import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.ICP import ICP
from utils.submap import SubmapManager
from utils.keyframe import KeyframeManager
from utils.posegraph import PoseGraph
from utils.loopclosure import LoopClosure
from utils.motion_predictor import MotionPredictor
from utils.io import load_folder, load_pcd
from utils.viz import Visualizer


def distribute_pose_corrections(poses, kf_indices, kf_opt_poses):
    new_poses = [p.copy() for p in poses]
    for k in range(len(kf_indices) - 1):
        idx_A, idx_B   = kf_indices[k], kf_indices[k + 1]
        orig_A, orig_B = poses[idx_A], poses[idx_B]
        opt_A,  opt_B  = kf_opt_poses[k], kf_opt_poses[k + 1]
        corr_A = opt_A @ np.linalg.inv(orig_A)
        corr_B = opt_B @ np.linalg.inv(orig_B)
        rot_A,  trans_A = R.from_matrix(corr_A[:3, :3]), corr_A[:3, 3]
        rot_B,  trans_B = R.from_matrix(corr_B[:3, :3]), corr_B[:3, 3]
        key_rots = R.from_quat(np.vstack([rot_A.as_quat(), rot_B.as_quat()]))
        try:
            slerp = Slerp([0, 1], key_rots)
        except Exception:
            slerp = None
        for j in range(idx_A, idx_B):
            t       = (j - idx_A) / float(idx_B - idx_A)
            rot_t   = slerp(t).as_matrix() if slerp else rot_A.as_matrix()
            trans_t = trans_A + t * (trans_B - trans_A)
            corr_j  = np.eye(4)
            corr_j[:3, :3] = rot_t
            corr_j[:3,  3] = trans_t
            new_poses[j]   = corr_j @ poses[j]
    if len(kf_indices) > 0:
        last_kf_idx = kf_indices[-1]
        last_corr   = kf_opt_poses[-1] @ np.linalg.inv(poses[last_kf_idx])
        for j in range(last_kf_idx, len(poses)):
            new_poses[j] = last_corr @ poses[j]
    return new_poses


target = "/home/uc/docker/self-drivingCars/catkin_ws/src/slam/slam_lidar/data/Track3"
folder = f"{target}/data/raw_pcd/"
files  = load_folder(folder)

optimize_every = 10

icp        = ICP()
submap_mgr = SubmapManager()
kf_manager = KeyframeManager()
pg         = PoseGraph()
lc         = LoopClosure(icp)
viz        = Visualizer()

predictor = MotionPredictor()

pg.add_prior()

prev_pose    = np.eye(4)
prev_delta_T = np.eye(4)
poses        = []
timestamps   = []
score        = np.inf

full_map      = np.zeros((0, 3))
kf_indices    = []
kf_poses_list = []
last_kf_pose  = np.eye(4)
current_kf_idx = 0

for i, f in enumerate(files):
    pts = load_pcd(f)
    if pts.shape[1] > 3:
        pts = pts[:, :3]

    if i == 0:
        poses.append(prev_pose)
        timestamps.append(f.split("/")[-1].replace(".pcd", ""))
        submap_mgr.add_keyframe(pts, prev_pose)
        lc.add_keyframe(pts, prev_pose)
        kf_indices.append(i)
        kf_poses_list.append(prev_pose.copy())
        continue

    cos_angle      = np.clip((np.trace(prev_delta_T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    turn_angle_deg = np.degrees(np.arccos(cos_angle))
    submap_window  = 5 if turn_angle_deg > 2.5 else None

    submap_pts = submap_mgr.get_latest_submap(window=submap_window)
    if submap_pts is None:
        continue

    init_T = predictor.predict(prev_pose)

    T, score = icp.align(pts, submap_pts, init_T)
    curr_pose = T.copy()

    # ICP correction magnitude from init_T — used as alignment confidence proxy.
    # Small delta → confident alignment → tight odometry edge.
    # Large delta → uncertain or large correction → loose edge.
    if np.isfinite(score):
        _dT_init  = np.linalg.inv(init_T) @ T
        icp_delta = float(np.linalg.norm(_dT_init[:3, 3]))
    else:
        icp_delta = np.inf
    odom_sigma = float(np.clip(0.05 + icp_delta * 0.08, 0.05, 0.30))

    poses.append(curr_pose)
    timestamps.append(f.split("/")[-1].replace(".pcd", ""))

    if np.isfinite(score):
        delta_T = np.linalg.inv(prev_pose) @ curr_pose
        predictor.update(delta_T, icp_score=score)
        prev_delta_T = delta_T.copy()

    icp_ok = np.isfinite(score)
    if icp_ok and kf_manager.is_keyframe(curr_pose):
        current_kf_idx += 1

        submap_mgr.add_keyframe(pts, curr_pose)
        lc.add_keyframe(pts, curr_pose)

        rel_pose = np.linalg.inv(last_kf_pose) @ curr_pose
        pg.add_node(current_kf_idx, curr_pose)
        pg.add_odom(current_kf_idx - 1, current_kf_idx, rel_pose, sigma=odom_sigma)

        last_kf_pose = curr_pose.copy()
        kf_indices.append(i)

        pts_w   = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
        full_map = np.concatenate([full_map, pts_w], axis=0)
        pcd_tmp  = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(full_map)
        pcd_tmp  = pcd_tmp.voxel_down_sample(voxel_size=0.5)
        full_map = np.asarray(pcd_tmp.points)

        should_optimize = False
        loop_result     = lc.detect(submap_mgr)
        if loop_result is not None:
            loop_idx, T_loop, _ = loop_result
            if (T_loop.shape == (4, 4)
                    and np.isfinite(T_loop).all()
                    and abs(np.linalg.det(T_loop[:3, :3]) - 1.0) < 0.05):
                pg.add_loop(loop_idx, current_kf_idx, T_loop)
                should_optimize = True

        if not should_optimize and current_kf_idx % optimize_every == 0:
            should_optimize = True

        if should_optimize:
            kf_opt_dict   = pg.optimize()
            kf_opt_poses  = [kf_opt_dict[k] for k in sorted(kf_opt_dict.keys())]

            submap_mgr.update_poses(kf_opt_poses)
            lc.update_poses(kf_opt_poses)

            poses         = distribute_pose_corrections(poses, kf_indices, kf_opt_poses)
            curr_pose     = kf_opt_poses[-1].copy()
            last_kf_pose  = curr_pose.copy()
            kf_poses_list = kf_opt_poses

            predictor.reset()

    prev_pose = curr_pose.copy()

    pts_curr_w = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
    viz.update(pts_curr_w, full_map, poses, kf_poses_list)

header = "timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33"
rows   = [[str(t)] + T.reshape(-1).tolist() for t, T in zip(timestamps, poses)]
df     = pd.DataFrame(rows, columns=header.split(','))
df.to_csv(f"{target}/result/lidar_poses.csv", index=False)
print(f"[Done] Saved {len(poses)} poses → {target}/result/lidar_poses.csv")