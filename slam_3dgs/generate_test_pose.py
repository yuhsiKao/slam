import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from pathlib import Path
from tqdm import tqdm

def load_and_interpolate(data_dir=None, track_dir=None, track_num=1):
    base = data_dir or "/home/uc/docker/self-drivingCars/catkin_ws/src/slam/slam_lidar/itri58_colored_pcd_t1"
    tdir = track_dir or "/home/uc/docker/self-drivingCars/catkin_ws/src/slam/slam_3dgs/track1"
    # --- PATHS ---
    IMG_DIR = os.path.join(base, "itri58_image")
    GT_POSE_FILE = os.path.join(base, "camera_gt_pose.txt")
    TEST_LIST_FILE = os.path.join(tdir, f"track{track_num}_test_frame_list.txt")
    OUTPUT_FILE = os.path.join(tdir, "test_pose_list.txt")

    # 1. Get all available timestamps from image filenames
    # Sorted ensures we match the order of the GT pose file lines
    all_img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
    all_timestamps = np.array([float(f.split('.')[0]) for f in all_img_files])
    
    # 2. Load all 1314 GT poses[cite: 5]
    # Use comma delimiter based on your snippet
    raw_gt = np.loadtxt(GT_POSE_FILE, delimiter=',') 
    num_poses = raw_gt.shape[0]
    
    if len(all_timestamps) != num_poses:
        print(f"[Warning] Image count ({len(all_timestamps)}) != Pose count ({num_poses})")
        # We take the minimum to be safe
        min_count = min(len(all_timestamps), num_poses)
        all_timestamps = all_timestamps[:min_count]
        raw_gt = raw_gt[:min_count]

    # 3. Decompose matrices into Position and Rotation (Quaternion)
    all_pos = []
    all_quat = []
    
    for i in range(len(raw_gt)):
        matrix = raw_gt[i].reshape(4, 4)
        all_pos.append(matrix[:3, 3])
        # Convert 3x3 rotation to quaternion for Slerp
        rot_mat = matrix[:3, :3]
        all_quat.append(R.from_matrix(rot_mat).as_quat())

    all_pos = np.array(all_pos)
    all_quat = np.array(all_quat)

    # 4. Setup Interpolation (Normalize timestamps to prevent precision loss)
    t0 = all_timestamps[0]
    norm_times = all_timestamps - t0
    
    pos_interp = interp1d(norm_times, all_pos, axis=0, kind='linear', fill_value="extrapolate")
    rotations = R.from_quat(all_quat)
    slerp = Slerp(norm_times, rotations)

    # 5. Process Test Frame List
    with open(TEST_LIST_FILE, 'r') as f:
        test_frames = [line.strip().split('.')[0] for line in f if line.strip()]
    
    interpolated_results = []
    
    print(f"Interpolating poses for {len(test_frames)} test frames...")
    for frame_id in test_frames:
        t_target = float(frame_id) - t0
        
        # Estimate Position and Rotation
        p_interp = pos_interp(t_target)
        r_interp = slerp(t_target).as_matrix()
        
        # Reconstruct 4x4 Matrix
        new_matrix = np.eye(4)
        new_matrix[:3, :3] = r_interp
        new_matrix[:3, 3] = p_interp
        
        interpolated_results.append(new_matrix.flatten())

    # 6. Save as flat 16-float lines for the evaluation script
    np.savetxt(OUTPUT_FILE, interpolated_results, fmt='%.18e')
    print(f"Successfully saved test poses to {OUTPUT_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--track_dir", default=None)
    parser.add_argument("--track_num", type=int, default=1)
    args = parser.parse_args()
    load_and_interpolate(args.data_dir, args.track_dir, args.track_num)