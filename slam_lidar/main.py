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
from utils.io import load_folder, load_pcd
from utils.viz import Visualizer

def distribute_pose_corrections(poses, kf_indices, kf_opt_poses):
    """
    Interpolate pose graph corrections to all non-keyframe poses using SLERP.
    """
    new_poses = [p.copy() for p in poses]
    
    for k in range(len(kf_indices) - 1):
        idx_A = kf_indices[k]
        idx_B = kf_indices[k+1]
        
        orig_A, orig_B = poses[idx_A], poses[idx_B]
        opt_A, opt_B = kf_opt_poses[k], kf_opt_poses[k+1]
        
        # Calculate correction transforms
        corr_A = opt_A @ np.linalg.inv(orig_A)
        corr_B = opt_B @ np.linalg.inv(orig_B)
        
        rot_A = R.from_matrix(corr_A[:3, :3])
        rot_B = R.from_matrix(corr_B[:3, :3])
        trans_A = corr_A[:3, 3]
        trans_B = corr_B[:3, 3]
        
        # Setup SLERP
        key_rots = R.from_quat(np.vstack([rot_A.as_quat(), rot_B.as_quat()]))
        try:
            slerp = Slerp([0, 1], key_rots)
        except Exception:
            slerp = None 
            
        # Apply interpolated correction to intermediate frames
        for j in range(idx_A, idx_B):
            t = (j - idx_A) / float(idx_B - idx_A)
            
            rot_t = slerp(t).as_matrix() if slerp else rot_A.as_matrix()
            trans_t = trans_A + t * (trans_B - trans_A)
            
            corr_j = np.eye(4)
            corr_j[:3, :3] = rot_t
            corr_j[:3, 3] = trans_t
            
            new_poses[j] = corr_j @ poses[j]
            
    # Apply the last correction to any trailing frames
    if len(kf_indices) > 0:
        last_kf_idx = kf_indices[-1]
        last_corr = kf_opt_poses[-1] @ np.linalg.inv(poses[last_kf_idx])
        for j in range(last_kf_idx, len(poses)):
            new_poses[j] = last_corr @ poses[j]
            
    return new_poses


# ==========================================
# Main SLAM Pipeline
# ==========================================
'''
Path setting
'''

target = "/home/uc/docker/self-drivingCars/catkin_ws/src/slam/slam_lidar/data/Track1"
folder = f"{target}/data/raw_pcd/"
files = load_folder(folder)

# Initialization
optimize_every = 20   # run pose graph optimization every N keyframes

icp = ICP()
submap_mgr = SubmapManager()
kf_manager = KeyframeManager()
pg = PoseGraph()
lc = LoopClosure(icp)
viz = Visualizer()

pg.add_prior()

prev_pose = np.eye(4)
prev_delta_T = np.eye(4)   # constant-velocity motion model: last relative motion
poses = []
timestamps = []

# Visualization & Interpolation state
full_map = np.zeros((0, 3))
kf_indices = []
kf_poses_list = []
last_kf_pose = np.eye(4)
current_kf_idx = 0

# main loop
for i, f in enumerate(files):
    pts = load_pcd(f)
    if pts.shape[1] > 3:
        pts = pts[:, :3]

    # Initial frame
    if i == 0:
        poses.append(prev_pose)
        timestamps.append(f.split("/")[-1].replace(".pcd", ""))
        submap_mgr.add_keyframe(pts, prev_pose)
        # Assuming updated LC to accept (local_pts, pose, world_submap)
        lc.add_keyframe(pts, prev_pose)
        
        kf_indices.append(i)
        kf_poses_list.append(prev_pose.copy())
        continue

    # ===== Adaptive Submap Window (turn detection) =====
    # Extract per-frame rotation angle from the motion model
    cos_angle = np.clip((np.trace(prev_delta_T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    turn_angle_deg = np.degrees(np.arccos(cos_angle))
    # if turn_angle_deg > 1.0:
    #     print(f"[Frame {i}] ---------------------------Turn angle: {turn_angle_deg:.2f} degrees")
    submap_window = 5 if turn_angle_deg > 2.5 else None  # wider window during turns

    submap_pts = submap_mgr.get_latest_submap(window=submap_window)
    if submap_pts is None:
        continue

    # ===== Scan-to-Submap (Odometry) =====
    init_T = prev_pose @ prev_delta_T  # constant-velocity prediction
    # init_T = prev_pose  #
    T, score = icp.align(pts, submap_pts, init_T)
    curr_pose = T.copy()
    
    poses.append(curr_pose)
    timestamps.append(f.split("/")[-1].replace(".pcd", ""))

    # ===== Keyframe Processing =====
    if kf_manager.is_keyframe(curr_pose):
        current_kf_idx += 1
        
        # 1. Add Node to Managers
        submap_mgr.add_keyframe(pts, curr_pose)
        lc.add_keyframe(pts, curr_pose)

        
        # 2. Add Odom Edge between Keyframes
        rel_pose = np.linalg.inv(last_kf_pose) @ curr_pose
        pg.add_node(current_kf_idx, curr_pose)
        pg.add_odom(current_kf_idx - 1, current_kf_idx, rel_pose)

        last_kf_pose = curr_pose.copy()
        kf_indices.append(i)
        
        # 3. Update Visual Map (Voxel downsampled)
        pts_w = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
        full_map = np.concatenate([full_map, pts_w], axis=0)
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(full_map)
        pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=0.5) 
        full_map = np.asarray(pcd_tmp.points)

        # ===== Loop Closure =====
        should_optimize = False
        loop_result = lc.detect(submap_mgr)
        if loop_result is not None:
            loop_idx, T_loop, score = loop_result
            pg.add_loop(loop_idx, current_kf_idx, T_loop)
            should_optimize = True

        # ===== Periodic Optimization =====
        if not should_optimize and current_kf_idx % optimize_every == 0:
            should_optimize = False

        # ===== Optimize & Interpolate =====
        if should_optimize:
            kf_opt_dict = pg.optimize()
            kf_opt_poses = [kf_opt_dict[k] for k in sorted(kf_opt_dict.keys())]

            submap_mgr.update_poses(kf_opt_poses)
            lc.update_poses(kf_opt_poses)

            poses = distribute_pose_corrections(poses, kf_indices, kf_opt_poses)
            curr_pose = kf_opt_poses[-1].copy()
            last_kf_pose = curr_pose.copy()
            kf_poses_list = kf_opt_poses

    if np.isfinite(score):                              # constant-velocity update
        prev_delta_T = np.linalg.inv(prev_pose) @ curr_pose
    prev_pose = curr_pose.copy()

    # ===== Visualization =====
    pts_curr_w = (curr_pose[:3, :3] @ pts.T).T + curr_pose[:3, 3]
    viz.update(pts_curr_w, full_map, poses, kf_poses_list)

# ===== Save Output CSV =====
header = "timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33"
rows = [[str(t)] + T.reshape(-1).tolist() for t, T in zip(timestamps, poses)]

df = pd.DataFrame(rows, columns=header.split(','))
df.to_csv(f"{target}/result/lidar_poses.csv", index=False)