import numpy as np

class SubmapManager:
    def __init__(self, window_size=3, submap_voxel=0.05):
        self.window       = window_size
        self.submap_voxel = submap_voxel

        self.keyframes_pts = []
        self.keyframes_pose = []
        self.submaps = []

    def add_keyframe(self, pts, pose):
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        self.keyframes_pts.append(pts)
        self.keyframes_pose.append(pose)
        submap = self.build_submap(len(self.keyframes_pts)-1)

        self.submaps.append(submap)

    def update_poses(self, new_kf_poses, change_thr: float = 0.05):
        """
        Update keyframe poses and incrementally rebuild only affected submaps.

        A submap at index i covers keyframes [i-window, i+window].
        It is rebuilt only when at least one keyframe in that range moved by
        more than `change_thr` metres (translation) or equivalent rotation,
        avoiding a full O(N) rebuild after every pose-graph optimisation.
        """
        N = len(self.keyframes_pts)

        # Fall back to full rebuild if pose count changed (e.g. first call)
        if len(new_kf_poses) != N:
            self.keyframes_pose = [p.copy() for p in new_kf_poses]
            self.submaps = [self.build_submap(i) for i in range(N)]
            return

        # 1. Identify keyframes whose pose changed significantly
        changed = np.zeros(N, dtype=bool)
        for i, (old_p, new_p) in enumerate(zip(self.keyframes_pose, new_kf_poses)):
            dt = np.linalg.norm(new_p[:3, 3] - old_p[:3, 3])
            dR = np.linalg.norm(new_p[:3, :3] - old_p[:3, :3], 'fro')
            changed[i] = (dt > change_thr) or (dR > change_thr)

        # 2. Update stored poses first
        self.keyframes_pose = [p.copy() for p in new_kf_poses]

        # 3. Rebuild only submaps whose window overlaps with a changed keyframe
        w = self.window
        rebuilt = 0
        for i in range(N):
            lo = max(0, i - w)
            hi = min(N, i + w + 1)
            if changed[lo:hi].any():
                self.submaps[i] = self.build_submap(i)
                rebuilt += 1

        print(f"[SubmapMgr] Incremental rebuild: {rebuilt}/{N} submaps "
              f"({int(changed.sum())} keyframes moved > {change_thr}m)")
            
    @staticmethod
    def _voxel_down(pts, voxel_size):
        """Fast voxel downsampling: keep one point per voxel cell."""
        if len(pts) == 0:
            return pts
        keys = np.floor(pts / voxel_size).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return pts[idx]

    def build_submap(self, idx, window=None):
        """Stack keyframes around idx into a voxel-downsampled world-frame point cloud."""
        w = window if window is not None else self.window
        start = max(0, idx - w)
        end = min(len(self.keyframes_pts), idx + w + 1)

        pts_list = []
        for i in range(start, end):
            pts  = self.keyframes_pts[i]
            pose = self.keyframes_pose[i]
            pts_list.append((pose[:3, :3] @ pts.T).T + pose[:3, 3])

        combined = np.concatenate(pts_list, axis=0)
        return self._voxel_down(combined, self.submap_voxel)

    def get_latest_submap(self, window=None):
        if len(self.submaps) == 0:
            return None
        # Use cached submap when window matches default; rebuild on-the-fly otherwise
        if window is None or window == self.window:
            return self.submaps[-1]
        return self.build_submap(len(self.keyframes_pts) - 1, window=window)