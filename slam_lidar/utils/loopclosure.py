import numpy as np


class LoopClosure:
    def __init__(
        self,
        icp,
        dist_thresh=8.0,
        fitness_thresh=0.25,
        max_candidates=20,
        min_separation=15,
        max_loop_correction=5.0,
    ):
        self.icp = icp
        self.dist_thresh = dist_thresh
        self.fitness_thresh = fitness_thresh
        self.max_candidates = max_candidates
        self.min_separation = min_separation
        self.max_loop_correction = max_loop_correction

        self.keyframes = []
        self.poses = []

    # =========================
    #       Add Keyframe
    # =========================
    def add_keyframe(self, pts, pose):
        self.keyframes.append(pts)
        self.poses.append(np.asarray(pose, dtype=np.float64).copy())

    # ==========================
    #    Candidate Selection
    # ==========================
    def find_candidates(self, curr_idx):
        candidates = []

        if curr_idx <= self.min_separation:
            return candidates

        curr_xyz = self.poses[curr_idx][:3, 3]
        search_end = curr_idx - self.min_separation

        candidate_info = []
        for i in range(search_end):
            dist = np.linalg.norm(curr_xyz - self.poses[i][:3, 3])
            if dist < self.dist_thresh:
                candidate_info.append((dist, i))

        # Sort by distance, keep closest max_candidates
        candidate_info.sort(key=lambda x: x[0])
        candidates = [i for _, i in candidate_info[:self.max_candidates]]

        return candidates

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

            # Reject if the loop correction is geometrically unreasonable
            correction = np.linalg.norm(T_abs[:3, 3] - curr_pose[:3, 3])
            if correction > self.max_loop_correction:
                print(f"[LoopClosure] Reject candidate={idx}, score={score:.4f}, correction={correction:.3f}m too large", flush=True)
                continue

            if score < best_score:
                best_score = score
                best_idx = idx
                best_T_abs = T_abs.copy()

        if best_idx is None:
            return None

        if best_score >= self.fitness_thresh:
            print(f"[LoopClosure] Reject best: curr={curr_idx}, candidate={best_idx}, score={best_score:.4f}", flush=True)
            return None

        # Relative pose: candidate_kf → current_kf (pose graph convention)
        T_loop = np.linalg.inv(self.poses[best_idx]) @ best_T_abs
        print(f"[LoopClosure] Found loop: curr={curr_idx}, candidate={best_idx}, score={best_score:.4f}", flush=True)
        return best_idx, T_loop, best_score

    # =========================
    #        Pose Update
    # =========================
    def update_poses(self, new_kf_poses):
        for i in range(len(new_kf_poses)):
            if i < len(self.poses):
                self.poses[i] = np.asarray(new_kf_poses[i], dtype=np.float64).copy()

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