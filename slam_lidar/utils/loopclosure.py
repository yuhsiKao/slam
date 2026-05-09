import numpy as np
from scipy.spatial import KDTree


class LoopClosure:
    def __init__(
        self,
        icp,
        dist_thresh=8.0,
        fitness_thresh=0.25,
        max_candidates=20,
        min_separation=15,
        max_loop_correction=5.0,
        confirm_thresh=2,           # consecutive detections needed before accepting
    ):
        self.icp = icp
        self.dist_thresh = dist_thresh
        self.fitness_thresh = fitness_thresh
        self.max_candidates = max_candidates
        self.min_separation = min_separation
        self.max_loop_correction = max_loop_correction
        self.confirm_thresh = confirm_thresh

        self.keyframes = []
        self.poses = []

        # KD-tree over keyframe XY positions (rebuilt incrementally)
        self._kd_tree = None
        self._kd_pts = None         # (N, 3) array backing the tree
        self._kd_dirty = False

        # Consecutive-confirmation buffer: candidate_idx -> hit count
        self._confirm_buf: dict[int, int] = {}

    # =========================
    #       Add Keyframe
    # =========================
    def add_keyframe(self, pts, pose):
        self.keyframes.append(pts)
        self.poses.append(np.asarray(pose, dtype=np.float64).copy())
        self._kd_dirty = True

    # ------------------------------------------------------------------
    # Adaptive search radius: widen during indoor/outdoor transitions
    # ------------------------------------------------------------------
    def _search_radius(self):
        env = self.icp.get_environment() if hasattr(self.icp, 'get_environment') else 'outdoor'
        # Indoor: tighter geometry → smaller radius is fine;
        # outdoor or transitioning: allow wider search to catch long-distance loops
        return self.dist_thresh if env == 'indoor' else self.dist_thresh * 1.5

    def _correction_limit(self):
        env = self.icp.get_environment() if hasattr(self.icp, 'get_environment') else 'outdoor'
        # Indoor/outdoor transitions can produce large corrections; be more lenient
        return self.max_loop_correction if env == 'indoor' else self.max_loop_correction * 2.5

    # ==========================
    #    Candidate Selection  (O(log n) via KD-tree)
    # ==========================
    def _rebuild_kdtree(self):
        if not self._kd_dirty or len(self.poses) == 0:
            return
        self._kd_pts = np.array([p[:3, 3] for p in self.poses])
        self._kd_tree = KDTree(self._kd_pts)
        self._kd_dirty = False

    def find_candidates(self, curr_idx):
        if curr_idx <= self.min_separation:
            return []

        self._rebuild_kdtree()

        curr_xyz = self.poses[curr_idx][:3, 3]
        radius = self._search_radius()
        indices = self._kd_tree.query_ball_point(curr_xyz, radius)

        # Keep only temporally separated frames
        cutoff = curr_idx - self.min_separation
        candidates = [i for i in indices if i < cutoff]
        candidates.sort(key=lambda i: np.linalg.norm(curr_xyz - self._kd_pts[i]))
        return candidates[:self.max_candidates]

    # =========================
    #       ICP Matching
    # =========================
    def match(self, curr_idx, candidates, submap_mgr):
        if not candidates:
            return None

        best_score = np.inf
        best_idx = None
        best_T_abs = None

        src_pts = self.keyframes[curr_idx]
        curr_pose = self.poses[curr_idx]
        corr_limit = self._correction_limit()

        for idx in candidates:
            try:
                tgt_pts = submap_mgr.submaps[idx]
            except Exception:
                continue

            if tgt_pts is None or len(tgt_pts) < 50:
                continue

            T_abs, score = self.icp.align(src_pts, tgt_pts, curr_pose)

            if T_abs is None or not np.isfinite(score) or not np.isfinite(T_abs).all():
                continue

            correction = np.linalg.norm(T_abs[:3, 3] - curr_pose[:3, 3])
            if correction > corr_limit:
                print(f"[LoopClosure] Reject candidate={idx}, score={score:.4f}, correction={correction:.3f}m > limit={corr_limit:.1f}m", flush=True)
                continue

            if score < best_score:
                best_score = score
                best_idx = idx
                best_T_abs = T_abs.copy()

        if best_idx is None:
            # decay confirmation counts for all candidates that weren't best
            self._confirm_buf.clear()
            return None

        if best_score >= self.fitness_thresh:
            print(f"[LoopClosure] Reject best: curr={curr_idx}, candidate={best_idx}, score={best_score:.4f}", flush=True)
            self._confirm_buf.clear()
            return None

        # Consecutive confirmation gate: require confirm_thresh consecutive hits
        self._confirm_buf[best_idx] = self._confirm_buf.get(best_idx, 0) + 1
        hits = self._confirm_buf[best_idx]
        # Remove stale entries (other candidates)
        self._confirm_buf = {best_idx: hits}

        if hits < self.confirm_thresh:
            print(f"[LoopClosure] Pending confirmation: curr={curr_idx}, candidate={best_idx}, score={best_score:.4f}, hits={hits}/{self.confirm_thresh}", flush=True)
            return None

        # Confirmed — reset counter so repeated loops are re-confirmed
        self._confirm_buf[best_idx] = 0

        T_loop = np.linalg.inv(self.poses[best_idx]) @ best_T_abs
        print(f"[LoopClosure] CONFIRMED loop: curr={curr_idx}, candidate={best_idx}, score={best_score:.4f}", flush=True)
        return best_idx, T_loop, best_score

    # =========================
    #        Pose Update
    # =========================
    def update_poses(self, new_kf_poses):
        for i in range(len(new_kf_poses)):
            if i < len(self.poses):
                self.poses[i] = np.asarray(new_kf_poses[i], dtype=np.float64).copy()
        self._kd_dirty = True   # positions changed → invalidate KD-tree

    # =========================
    #            API
    # =========================
    def detect(self, submap_mgr):
        curr_idx = len(self.keyframes) - 1

        if curr_idx < self.min_separation:
            return None

        candidates = self.find_candidates(curr_idx)
        if candidates:
            print(f"[LoopClosure] curr={curr_idx}, candidates={candidates}", flush=True)

        return self.match(curr_idx, candidates, submap_mgr)