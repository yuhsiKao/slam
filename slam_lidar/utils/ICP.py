import small_gicp as sg
import numpy as np

# Indoor: p95 of range distribution must be below this threshold (metres)
_INDOOR_RANGE_P95 = 40.0

# Per-environment ICP tuning
# min_z / max_z apply to both source (sensor frame) and target (world frame).
# Indoor: tight Z band removes noise above ceiling and below floor,
#         giving GICP a stable horizontal-surface constraint for Z estimation.
_ENV_PARAMS = {
    'indoor':  dict(voxel_size=0.20, max_correspondence_distance=1.0,
                    max_range=30.0,  max_iterations=50,  registration_type='GICP',
                    min_z=-3.0, max_z=10.0, max_delta_trans =1.5),
    'outdoor': dict(voxel_size=0.30, max_correspondence_distance=1.5,                    max_range=100.0, max_iterations=80,  registration_type='GICP',
                    min_z=-3.0, max_z=30.0, max_delta_trans =4.0),
}


class ICP:
    def __init__(
        self,
        num_threads=4,
        num_neighbors=20,
        max_delta_trans=5.0,
        min_points=50,
        min_range=0.5,
        min_z=-3.0,
        max_z=30.0,
    ):
        self.num_threads     = num_threads
        self.num_neighbors   = num_neighbors
        self.max_delta_trans = max_delta_trans
        self.max_delta_rot   = 30.0   # degrees — reject if ICP rotates > 30° from init_T
        self.min_points      = min_points
        self.min_range       = min_range
        self.min_z           = min_z
        self.max_z           = max_z

        # early stopping — stop when BOTH translation AND rotation delta are tiny
        self.early_stop_delta = 1e-4   # metres
        self.early_stop_rot   = 1e-4   # radians (~0.006°)
        self.early_stop_chunk = 5      # iterations per convergence check

        # Asymmetric hysteresis: switch to indoor fast, back to outdoor conservatively.
        # Prevents drift at corners caused by coarse outdoor params during transition.
        self._env             = 'outdoor'
        self._vote_window_in  = 1   # 1 consecutive indoor vote  → switch to indoor
        self._vote_window_out = 2   # 2 consecutive outdoor votes → switch back to outdoor
        self._vote_buffer     = []

    # ------------------------------------------------------------------
    # Environment detection
    # ------------------------------------------------------------------
    def _detect_environment(self, pts):
        """Classify indoor/outdoor from raw range distribution with asymmetric hysteresis."""
        r = np.linalg.norm(pts[:, :3], axis=1)
        r = r[(r > self.min_range) & np.isfinite(r)]
        if len(r) < 10:
            return self._env
        p95  = np.percentile(r, 95)
        vote = 'indoor' if p95 < _INDOOR_RANGE_P95 else 'outdoor'
        print(f"[ICP] env_detect: p95={p95:.1f}m  n_pts={len(r)}  vote={vote}  current={self._env}", flush=True)

        self._vote_buffer.append(vote)
        max_win = max(self._vote_window_in, self._vote_window_out)
        if len(self._vote_buffer) > max_win:
            self._vote_buffer.pop(0)

        required = self._vote_window_in if vote == 'indoor' else self._vote_window_out
        if (len(self._vote_buffer) >= required
                and all(v == vote for v in self._vote_buffer[-required:])
                and vote != self._env):
            print(f"[ICP] Environment → {vote.upper()}  (p95={p95:.1f}m)", flush=True)
            self._env = vote

        return self._env

    def get_environment(self):
        return self._env

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def _filter(self, pts, max_range=100.0, min_z=None, max_z=None):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Expected Nx3 point cloud, got {pts.shape}")
        pts = pts[:, :3]
        pts = pts[np.isfinite(pts).all(axis=1)]
        r   = np.linalg.norm(pts, axis=1)
        pts = pts[(r > self.min_range) & (r < max_range)]
        z_lo = min_z if min_z is not None else self.min_z
        z_hi = max_z if max_z is not None else self.max_z
        pts = pts[(pts[:, 2] > z_lo) & (pts[:, 2] < z_hi)]
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

        ep         = _ENV_PARAMS[env]
        voxel_size = ep['voxel_size']
        max_corr   = ep['max_correspondence_distance']
        max_range  = ep['max_range']
        max_iter   = ep['max_iterations']
        reg_type   = ep['registration_type']
        min_z      = ep['min_z']
        max_z      = ep['max_z']

        # Source is in sensor frame → apply env-specific range + Z filter.
        # Target is the world-frame submap: its points are already bounded by
        # the submap window, so range must NOT be filtered by distance from world
        # origin (that distance grows as the robot moves and would eliminate all
        # nearby submap points once the robot is >max_range from the start).
        src_pts = self._filter(src_raw, max_range=max_range, min_z=min_z, max_z=max_z)
        tgt_pts = self._filter(np.asarray(tgt_pts, dtype=np.float64),
                               max_range=1e9,
                               min_z=self.min_z, max_z=self.max_z)

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

            # Guard: voxel downsampling may reduce point count below num_neighbors.
            # small_gicp's covariance estimation will segfault if points < num_neighbors.
            src_ds_n = source.size() if hasattr(source, 'size') else 0
            tgt_ds_n = target.size() if hasattr(target, 'size') else 0
            if src_ds_n < self.num_neighbors or tgt_ds_n < self.num_neighbors:
                return init_T.copy(), np.inf

            chunk  = self.early_stop_chunk
            T_curr = init_T.copy()
            result = None
            for _ in range(0, max_iter, chunk):
                result = sg.align(
                    target, source, target_tree,
                    init_T_target_source=T_curr,
                    registration_type=reg_type,
                    max_correspondence_distance=max_corr,
                    num_threads=self.num_threads,
                    max_iterations=chunk,
                )
                T_new       = np.asarray(result.T_target_source, dtype=np.float64)
                dT          = np.linalg.inv(T_curr) @ T_new
                trans_delta = np.linalg.norm(dT[:3, 3])
                cos_a       = np.clip((np.trace(dT[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
                rot_delta   = np.arccos(cos_a)
                T_curr      = T_new
                if trans_delta < self.early_stop_delta and rot_delta < self.early_stop_rot:
                    break

            T = T_curr

            n_ds  = max(source.size() if hasattr(source, 'size') else len(src_pts), 1)
            score = float(result.error) / n_ds

            if T.shape != (4, 4) or not np.isfinite(T).all() or not np.isfinite(score):
                return init_T.copy(), np.inf

            dT_from_init = np.linalg.inv(init_T) @ T
            delta_trans  = np.linalg.norm(dT_from_init[:3, 3])
            cos_a        = np.clip((np.trace(dT_from_init[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            delta_rot    = np.degrees(np.arccos(cos_a))

            if delta_trans > self.max_delta_trans:
                print(f"[ICP] rejected: delta_trans={delta_trans:.3f}m > {self.max_delta_trans}  env={env}", flush=True)
                return init_T.copy(), np.inf

            if delta_rot > self.max_delta_rot:
                print(f"[ICP] rejected: delta_rot={delta_rot:.1f}° > {self.max_delta_rot}  env={env}", flush=True)
                return init_T.copy(), np.inf

            print(f"[ICP] OK  score={score:.4f}  delta={delta_trans:.4f}m  rot={delta_rot:.2f}°  env={env}  reg={reg_type}", flush=True)
            return T, score

        except Exception as e:
            print(f"[ICP] align failed: {e}", flush=True)
            return init_T.copy(), np.inf
