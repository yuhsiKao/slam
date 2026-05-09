import small_gicp as sg
import numpy as np

# Indoor: p95 range < this threshold (metres)
_INDOOR_RANGE_P95 = 15.0

# Per-environment ICP tuning
_ENV_PARAMS = {
    'indoor': dict(voxel_size=0.10, max_correspondence_distance=1.5, max_range=30.0),
    'outdoor': dict(voxel_size=0.25, max_correspondence_distance=4.0, max_range=100.0),
}


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

        # Environment state with hysteresis counter to suppress flickering
        self._env = 'outdoor'
        self._env_vote = 0          # positive → outdoor, negative → indoor
        self._env_hysteresis = 5    # consecutive frames needed to switch

    # ------------------------------------------------------------------
    # Environment detection
    # ------------------------------------------------------------------
    def _detect_environment(self, pts):
        """Classify indoor/outdoor from raw (pre-filter) range distribution."""
        r = np.linalg.norm(pts[:, :3], axis=1)
        r = r[(r > self.min_range) & np.isfinite(r)]
        if len(r) < 10:
            return self._env
        p95 = np.percentile(r, 95)
        vote = -1 if p95 < _INDOOR_RANGE_P95 else +1
        self._env_vote = np.clip(self._env_vote + vote, -self._env_hysteresis, self._env_hysteresis)
        if self._env_vote <= -self._env_hysteresis:
            if self._env != 'indoor':
                print(f"[ICP] Environment → INDOOR  (p95_range={p95:.1f}m)", flush=True)
            self._env = 'indoor'
        elif self._env_vote >= self._env_hysteresis:
            if self._env != 'outdoor':
                print(f"[ICP] Environment → OUTDOOR (p95_range={p95:.1f}m)", flush=True)
            self._env = 'outdoor'
        return self._env

    def get_environment(self):
        return self._env

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def _filter(self, pts, max_range=None):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Expected Nx3 point cloud, got {pts.shape}")
        pts = pts[:, :3]
        pts = pts[np.isfinite(pts).all(axis=1)]
        r = np.linalg.norm(pts, axis=1)
        mr = max_range if max_range is not None else self.max_range
        pts = pts[(r > self.min_range) & (r < mr)]
        pts = pts[(pts[:, 2] > self.min_z) & (pts[:, 2] < self.max_z)]
        return np.ascontiguousarray(pts, dtype=np.float64)

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------
    def align(self, src_pts, tgt_pts, init_T=np.eye(4)):
        src_raw = np.asarray(src_pts, dtype=np.float64)
        if src_raw.ndim == 2 and src_raw.shape[1] >= 3:
            env = self._detect_environment(src_raw)
        else:
            env = self._env

        ep = _ENV_PARAMS[env]
        voxel_size = ep['voxel_size']
        max_corr = ep['max_correspondence_distance']
        max_range = ep['max_range']

        src_pts = self._filter(src_raw, max_range=max_range)
        tgt_pts = self._filter(np.asarray(tgt_pts, dtype=np.float64), max_range=max_range)

        init_T = np.ascontiguousarray(np.asarray(init_T, dtype=np.float64))
        if init_T.shape != (4, 4):
            init_T = np.eye(4, dtype=np.float64)

        if len(src_pts) < self.min_points or len(tgt_pts) < self.min_points:
            return init_T.copy(), np.inf

        try:
            target, target_tree = sg.preprocess_points(
                tgt_pts,
                downsampling_resolution=voxel_size,
                num_neighbors=self.num_neighbors,
                num_threads=self.num_threads,
            )
            source, _ = sg.preprocess_points(
                src_pts,
                downsampling_resolution=voxel_size,
                num_neighbors=self.num_neighbors,
                num_threads=self.num_threads,
            )

            result = sg.align(
                target, source, target_tree,
                init_T_target_source=init_T,
                registration_type='GICP',
                max_correspondence_distance=max_corr,
                num_threads=self.num_threads,
                max_iterations=self.max_iterations,
            )

            T = np.asarray(result.T_target_source, dtype=np.float64)

            # Normalise by actual inlier count (not raw point count)
            num_inliers = int(result.num_inliers) if hasattr(result, 'num_inliers') and result.num_inliers > 0 else None
            if num_inliers is not None:
                score = float(result.error) / num_inliers
            else:
                # fallback: normalise by downsampled source size (safer than raw count)
                n_ds = max(source.size() if hasattr(source, 'size') else len(src_pts), 1)
                score = float(result.error) / n_ds

            if T.shape != (4, 4) or not np.isfinite(T).all() or not np.isfinite(score):
                return init_T.copy(), np.inf

            delta_trans = np.linalg.norm((np.linalg.inv(init_T) @ T)[:3, 3])
            if delta_trans > self.max_delta_trans:
                print(f"[ICP] rejected: delta_trans={delta_trans:.3f}m too large", flush=True)
                return init_T.copy(), np.inf

            if score > self.reject_score:
                print(f"[ICP] rejected: score={score:.4f} > {self.reject_score} env={env}", flush=True)
                return init_T.copy(), np.inf

            print(f"[ICP] OK  score={score:.4f}  inliers={num_inliers}  delta_trans={delta_trans:.4f}m  env={env}", flush=True)
            return T, score

        except Exception as e:
            print(f"[ICP] align failed: {e}", flush=True)
            return init_T.copy(), np.inf
