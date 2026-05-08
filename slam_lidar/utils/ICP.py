import small_gicp as sg
import numpy as np


class ICP:
    def __init__(
        self,
        voxel_size=0.25,
        max_correspondence_distance=4.0,
        num_threads=4,
        num_neighbors=20,
        max_iterations=50,
        reject_score=1.0,
        max_delta_trans=5.0,
        min_points=50,
        min_range=0.5,
        max_range=100.0,
        min_z=-10.0,
        max_z=30.0,
    ):
        self.voxel_size = voxel_size
        self.max_correspondence_distance = max_correspondence_distance
        self.num_threads = num_threads
        self.num_neighbors = num_neighbors
        self.max_iterations = max_iterations
        self.reject_score = reject_score
        self.max_delta_trans = max_delta_trans
        self.min_points = min_points
        self.min_range = min_range
        self.max_range = max_range
        self.min_z = min_z
        self.max_z = max_z

    def _filter(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Expected Nx3 point cloud, got {pts.shape}")
        pts = pts[:, :3]
        pts = pts[np.isfinite(pts).all(axis=1)]
        r = np.linalg.norm(pts, axis=1)
        pts = pts[(r > self.min_range) & (r < self.max_range)]
        pts = pts[(pts[:, 2] > self.min_z) & (pts[:, 2] < self.max_z)]
        return np.ascontiguousarray(pts, dtype=np.float64)

    def align(self, src_pts, tgt_pts, init_T=np.eye(4)):
        src_pts = self._filter(src_pts)
        tgt_pts = self._filter(tgt_pts)

        init_T = np.ascontiguousarray(np.asarray(init_T, dtype=np.float64))
        if init_T.shape != (4, 4):
            init_T = np.eye(4, dtype=np.float64)

        if len(src_pts) < self.min_points or len(tgt_pts) < self.min_points:
            return init_T.copy(), np.inf

        try:
            target, target_tree = sg.preprocess_points(
                tgt_pts,
                downsampling_resolution=self.voxel_size,
                num_neighbors=self.num_neighbors,
                num_threads=self.num_threads,
            )
            source, _ = sg.preprocess_points(
                src_pts,
                downsampling_resolution=self.voxel_size,
                num_neighbors=self.num_neighbors,
                num_threads=self.num_threads,
            )

            result = sg.align(
                target, source, target_tree,
                init_T_target_source=init_T,
                registration_type='GICP',
                max_correspondence_distance=self.max_correspondence_distance,
                num_threads=self.num_threads,
                max_iterations=self.max_iterations,
            )

            T = np.asarray(result.T_target_source, dtype=np.float64)
            num_inliers = int(result.num_inliers) if hasattr(result, 'num_inliers') else len(src_pts)
            score = float(result.error) / num_inliers if num_inliers > 0 else np.inf

            # Validity checks
            if T.shape != (4, 4) or not np.isfinite(T).all() or not np.isfinite(score):
                return init_T.copy(), np.inf

            delta_trans = np.linalg.norm((np.linalg.inv(init_T) @ T)[:3, 3])
            if delta_trans > self.max_delta_trans:
                print(f"[ICP] rejected: delta_trans={delta_trans:.3f}m too large", flush=True)
                return init_T.copy(), np.inf

            if score > self.reject_score:
                print(f"[ICP] rejected: score={score:.4f} > {self.reject_score}", flush=True)
                return init_T.copy(), np.inf

            print(f"[ICP] OK  score={score:.4f}  inliers={num_inliers}  delta_trans={delta_trans:.4f}m", flush=True)
            return T, score

        except Exception as e:
            print(f"[ICP] align failed: {e}", flush=True)
            return init_T.copy(), np.inf
