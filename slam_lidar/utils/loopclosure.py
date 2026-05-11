"""
loopclosure.py  (descriptor-enhanced version)
===============================================
改進摘要：在 ICP 之前加入三層過濾，大幅降低誤報、提高召回：

Pipeline 新舊對比
─────────────────────────────────────────────────────────────
舊版：空間 KD-tree 候選 → ICP (init = curr_pose) → confirm
新版：空間 KD-tree 候選
        → [NEW] FPFH 描述子相似度排序（過濾不像的場景）
        → [NEW] RANSAC 粗對齊（給 ICP 一個靠譜的 init_T）
        → ICP 精細化
        → correction limit + fitness gate
        → confirm_thresh
─────────────────────────────────────────────────────────────

FPFH (Fast Point Feature Histogram)：
  - 每個點用周圍鄰域的法向量角度分佈編碼局部幾何
  - 33-dim 直方圖，對旋轉敏感但對平移不敏感
  - 兩幀之間算描述子距離 → 快速判斷場景是否雷同
  - 使用 open3d，已在環境中安裝

RANSAC global registration：
  - 從 FPFH 相似的點對中隨機抽樣求解 4-DoF 對齊
  - 完全不需要初始值，能跨越幾十公尺的漂移
  - 輸出一個粗 T 給 ICP，讓 ICP 不再依賴 curr_pose 當初始值
"""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


# ──────────────────────────────────────────────────────
#  FPFH tuning — per-environment params
#  Indoor:  narrow corridors need finer resolution
#  Outdoor: open areas, coarser sampling is fine
# ──────────────────────────────────────────────────────
_FPFH_PARAMS = {
    'indoor':  dict(voxel=0.15, normal_r=0.5,  feature_r=1.0, ransac_iters=2000),
    'outdoor': dict(voxel=0.50, normal_r=1.5,  feature_r=3.0, ransac_iters=4000),
}
_DESC_DIST_THRESH  = 60.0   # FPFH L2 距離閾值：高於此值直接排除候選
_RANSAC_DIST       = 1.0    # RANSAC 對應點距離容忍 (m)
_RANSAC_CONF       = 0.95   # RANSAC 置信度
_FPFH_CACHE_MAX    = 150    # 最多快取幾幀的 FPFH（超過時淘汰最舊）


def _to_o3d(pts: np.ndarray, voxel: float, normal_r: float) -> o3d.geometry.PointCloud:
    """numpy Nx3 → open3d PointCloud，降採樣 + 法向量估算。"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return o3d.geometry.PointCloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd = pcd.voxel_down_sample(voxel_size=voxel)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_r, max_nn=30)
    )
    return pcd


def _compute_fpfh(pcd: o3d.geometry.PointCloud,
                  feature_r: float) -> o3d.pipelines.registration.Feature:
    """計算 FPFH 描述子（33-dim per point）。"""
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_r, max_nn=100)
    )


def _descriptor_distance(feat_a: o3d.pipelines.registration.Feature,
                          feat_b: o3d.pipelines.registration.Feature) -> float:
    """
    全域描述子相似度：取每個點的最近鄰距離的中位數。
    越小代表兩幀場景越相似。
    """
    fa = np.asarray(feat_a.data).T   # (N, 33)
    fb = np.asarray(feat_b.data).T   # (M, 33)

    if len(fa) == 0 or len(fb) == 0:
        return np.inf

    # 為了速度，只用 fa 的子集（最多 200 個點）
    if len(fa) > 200:
        idx = np.random.choice(len(fa), 200, replace=False)
        fa = fa[idx]

    # 對 fb 建 KD-tree（33 維），查每個 fa 點的最近鄰距離
    try:
        tree = KDTree(fb)
        dists, _ = tree.query(fa, k=1)
        return float(np.median(dists))
    except Exception:
        return np.inf


def _ransac_align(
    src_pcd: o3d.geometry.PointCloud,
    tgt_pcd: o3d.geometry.PointCloud,
    src_feat: o3d.pipelines.registration.Feature,
    tgt_feat: o3d.pipelines.registration.Feature,
    max_iters: int = 4000,
) -> np.ndarray:
    """
    FPFH + RANSAC 全域粗對齊。
    不需要初始值，從描述子對應關係直接求解 T。
    返回 4x4 变换矩阵，失敗時返回 None。
    """
    try:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd, tgt_pcd,
            src_feat, tgt_feat,
            mutual_filter=True,
            max_correspondence_distance=_RANSAC_DIST,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(_RANSAC_DIST),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iters, _RANSAC_CONF
            ),
        )
        T = np.asarray(result.transformation, dtype=np.float64)
        if T.shape == (4, 4) and np.isfinite(T).all() and abs(np.linalg.det(T[:3, :3]) - 1.0) < 0.05:
            return T
        return None
    except Exception as e:
        print(f"[LoopClosure.RANSAC] failed: {e}", flush=True)
        return None


class LoopClosure:
    def __init__(
        self,
        icp,
        dist_thresh=8.0,
        fitness_thresh=0.25,
        max_candidates=20,
        min_separation=15,
        max_loop_correction=2.5,
        max_loop_rotation=45.0,
        confirm_thresh=2,
        # ── 新增描述子參數 ──────────────────────────
        use_descriptors: bool = True,   # 是否啟用 FPFH 描述子篩選
        desc_top_k: int = 3,            # 描述子篩選後保留幾個候選進 RANSAC+ICP
        desc_dist_thresh: float = _DESC_DIST_THRESH,  # 超過此值排除
    ):
        self.icp = icp
        self.dist_thresh = dist_thresh
        self.fitness_thresh = fitness_thresh
        self.max_candidates = max_candidates
        self.min_separation = min_separation
        self.max_loop_correction = max_loop_correction
        self.max_loop_rotation   = max_loop_rotation
        self.confirm_thresh = confirm_thresh
        self.use_descriptors = use_descriptors
        self.desc_top_k = desc_top_k
        self.desc_dist_thresh = desc_dist_thresh

        self.keyframes = []     # raw pts (local frame)
        self.poses = []         # world poses

        # 描述子快取（lazy computed）
        self._fpfh_cache: dict[int, tuple] = {}   # idx → (pcd_ds, feat)

        # KD-tree over keyframe positions
        self._kd_tree = None
        self._kd_pts = None
        self._kd_dirty = False

        # 連續確認緩衝
        self._confirm_buf: dict[int, int] = {}

    # ====================================================
    #  Add Keyframe
    # ====================================================
    def add_keyframe(self, pts, pose):
        self.keyframes.append(pts)
        self.poses.append(np.asarray(pose, dtype=np.float64).copy())
        self._kd_dirty = True
        # 描述子 lazy：加入時不計算，等需要時再算（節省 add_keyframe 的時間）

    # ====================================================
    #  Environment-adaptive thresholds （與原版相同）
    # ====================================================
    def _search_radius(self):
        env = self.icp.get_environment() if hasattr(self.icp, 'get_environment') else 'outdoor'
        return self.dist_thresh if env == 'indoor' else self.dist_thresh * 1.5

    def _correction_limit(self):
        env = self.icp.get_environment() if hasattr(self.icp, 'get_environment') else 'outdoor'
        return self.max_loop_correction if env == 'indoor' else self.max_loop_correction * 2.5

    # ====================================================
    #  KD-tree rebuild
    # ====================================================
    def _rebuild_kdtree(self):
        if not self._kd_dirty or len(self.poses) == 0:
            return
        self._kd_pts = np.array([p[:3, 3] for p in self.poses])
        self._kd_tree = KDTree(self._kd_pts)
        self._kd_dirty = False

    # ====================================================
    #  Candidate Selection  （與原版相同）
    # ====================================================
    def find_candidates(self, curr_idx):
        if curr_idx <= self.min_separation:
            return []
        self._rebuild_kdtree()
        curr_xyz = self.poses[curr_idx][:3, 3]
        radius = self._search_radius()
        indices = self._kd_tree.query_ball_point(curr_xyz, radius)
        cutoff = curr_idx - self.min_separation
        candidates = [i for i in indices if i < cutoff]
        candidates.sort(key=lambda i: np.linalg.norm(curr_xyz - self._kd_pts[i]))
        return candidates[:self.max_candidates]

    # ====================================================
    #  [NEW] FPFH 描述子快取與計算
    # ====================================================
    def _env_fpfh_params(self) -> dict:
        """Return FPFH params for the current environment."""
        env = self.icp.get_environment() if hasattr(self.icp, 'get_environment') else 'outdoor'
        return _FPFH_PARAMS[env]

    def _get_fpfh(self, idx: int):
        """
        取得第 idx 個 keyframe 的 (pcd_downsampled, fpfh_feature)。
        快取以 (idx, env) 為 key；環境切換時自動重算。
        超過 _FPFH_CACHE_MAX 時淘汰最舊的 entry。
        """
        p   = self._env_fpfh_params()
        key = (idx, p['voxel'])

        if key in self._fpfh_cache:
            return self._fpfh_cache[key]

        pts = self.keyframes[idx]
        try:
            pcd = _to_o3d(pts, p['voxel'], p['normal_r'])
            if len(pcd.points) < 10:
                return None
            feat = _compute_fpfh(pcd, p['feature_r'])

            # Evict oldest entry when cache is full
            while len(self._fpfh_cache) >= _FPFH_CACHE_MAX:
                self._fpfh_cache.pop(next(iter(self._fpfh_cache)))

            self._fpfh_cache[key] = (pcd, feat)
            return (pcd, feat)
        except Exception as e:
            print(f"[LoopClosure.FPFH] idx={idx} failed: {e}", flush=True)
            return None

    # ====================================================
    #  [NEW] 描述子篩選：對候選列表打分並排序
    # ====================================================
    def _rank_by_descriptor(self, curr_idx: int, candidates: list) -> list:
        """
        對所有候選計算描述子距離，
        過濾掉距離超過閾值的，按距離升序返回前 desc_top_k 個。
        若描述子計算失敗，退回原始距離排序（保守策略）。
        """
        curr_data = self._get_fpfh(curr_idx)
        if curr_data is None:
            print("[LoopClosure] FPFH failed for current frame, skipping descriptor filter", flush=True)
            return candidates[:self.desc_top_k]

        _, curr_feat = curr_data

        scored = []
        for idx in candidates:
            cand_data = self._get_fpfh(idx)
            if cand_data is None:
                continue
            _, cand_feat = cand_data
            dist = _descriptor_distance(curr_feat, cand_feat)
            scored.append((idx, dist))
            print(f"[LoopClosure.FPFH] curr={curr_idx}, candidate={idx}, desc_dist={dist:.2f}", flush=True)

        # 過濾描述子距離太大的候選
        scored = [(i, d) for i, d in scored if d < self.desc_dist_thresh]

        if not scored:
            print(f"[LoopClosure.FPFH] all candidates rejected by descriptor threshold={self.desc_dist_thresh}", flush=True)
            return []

        scored.sort(key=lambda x: x[1])
        top = [i for i, _ in scored[:self.desc_top_k]]
        print(f"[LoopClosure.FPFH] top-{self.desc_top_k} after descriptor filter: {top}", flush=True)
        return top

    # ====================================================
    #  ICP Matching — 現在用 RANSAC init_T 取代 curr_pose
    # ====================================================
    def match(self, curr_idx, candidates, submap_mgr):
        if not candidates:
            return None

        # ── [NEW] Step 1: 描述子排序，大幅縮小候選集 ──────
        if self.use_descriptors and len(candidates) > 1:
            candidates = self._rank_by_descriptor(curr_idx, candidates)
        if not candidates:
            return None

        src_pts   = self.keyframes[curr_idx]
        curr_pose = self.poses[curr_idx]
        corr_limit = self._correction_limit()

        best_score   = np.inf
        best_idx     = None
        best_T_abs   = None

        for idx in candidates:
            try:
                tgt_pts = submap_mgr.submaps[idx]
            except Exception:
                continue
            if tgt_pts is None or len(tgt_pts) < 50:
                continue

            # ── [NEW] Step 2: RANSAC 粗對齊取得 init_T ────
            init_T = curr_pose   # fallback（保留原版行為）

            if self.use_descriptors:
                src_data = self._get_fpfh(curr_idx)
                cand_data = self._get_fpfh(idx)
                if src_data is not None and cand_data is not None:
                    src_pcd, src_feat = src_data
                    cand_pcd, cand_feat = cand_data
                    # 把 cand_pcd 轉換到世界座標（submap 是世界座標系）
                    cand_pcd_world = o3d.geometry.PointCloud(cand_pcd)
                    cand_pcd_world.transform(self.poses[idx])

                    T_ransac = _ransac_align(src_pcd, cand_pcd_world, src_feat, cand_feat,
                                             max_iters=self._env_fpfh_params()['ransac_iters'])
                    if T_ransac is not None:
                        # T_ransac 把 src 對齊到 world，作為 ICP 初始值
                        init_T = T_ransac
                        print(f"[LoopClosure.RANSAC] curr={curr_idx}, candidate={idx}: got coarse T", flush=True)
                    else:
                        print(f"[LoopClosure.RANSAC] curr={curr_idx}, candidate={idx}: failed, using curr_pose", flush=True)

            # ── Step 3: ICP 精細化（與原版相同，但 init_T 更好）
            T_abs, score = self.icp.align(src_pts, tgt_pts, init_T)

            if T_abs is None or not np.isfinite(score) or not np.isfinite(T_abs).all():
                continue

            correction = np.linalg.norm(T_abs[:3, 3] - curr_pose[:3, 3])
            if correction > corr_limit:
                print(f"[LoopClosure] Reject candidate={idx}, correction={correction:.3f}m > limit={corr_limit:.1f}m", flush=True)
                continue

            T_rel     = np.linalg.inv(curr_pose) @ T_abs
            cos_a     = np.clip((np.trace(T_rel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            rot_corr  = np.degrees(np.arccos(cos_a))
            rot_limit = self.max_loop_rotation if self.icp.get_environment() == 'indoor' else self.max_loop_rotation * 1.5
            if rot_corr > rot_limit:
                print(f"[LoopClosure] Reject candidate={idx}, rot_correction={rot_corr:.1f}° > limit={rot_limit:.1f}°", flush=True)
                continue

            if score < best_score:
                best_score = score
                best_idx   = idx
                best_T_abs = T_abs.copy()

        if best_idx is None:
            self._confirm_buf.clear()
            return None

        # ── 連續確認門 （與原版相同）──────────────────────
        self._confirm_buf[best_idx] = self._confirm_buf.get(best_idx, 0) + 1
        hits = self._confirm_buf[best_idx]
        self._confirm_buf = {best_idx: hits}

        if hits < self.confirm_thresh:
            print(f"[LoopClosure] Pending: curr={curr_idx}, candidate={best_idx}, score={best_score:.4f}, hits={hits}/{self.confirm_thresh}", flush=True)
            return None

        self._confirm_buf[best_idx] = 0
        T_loop = np.linalg.inv(self.poses[best_idx]) @ best_T_abs
        print(f"[LoopClosure] CONFIRMED loop: curr={curr_idx}, candidate={best_idx}, score={best_score:.4f}", flush=True)
        return best_idx, T_loop, best_score

    # ====================================================
    #  Pose Update  （與原版相同）
    # ====================================================
    def update_poses(self, new_kf_poses):
        for i in range(len(new_kf_poses)):
            if i < len(self.poses):
                self.poses[i] = np.asarray(new_kf_poses[i], dtype=np.float64).copy()
        self._kd_dirty = True
        # 注意：poses 更新後，舊的 FPFH 描述子仍然有效（描述子是局部特徵，不受世界座標變化影響）

    # ====================================================
    #  Public API  （與原版相同）
    # ====================================================
    def detect(self, submap_mgr):
        curr_idx = len(self.keyframes) - 1
        if curr_idx < self.min_separation:
            return None
        candidates = self.find_candidates(curr_idx)
        if candidates:
            print(f"[LoopClosure] curr={curr_idx}, spatial candidates={candidates}", flush=True)
        return self.match(curr_idx, candidates, submap_mgr)