import open3d as o3d
import pandas as pd
import numpy as np
import os
import os
import glob
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.spatial import cKDTree
from scipy.ndimage import minimum_filter

# --- camera Intrinsics  ---

'''
Those are for Track3
'''
W, H = 960, 720
FX, FY = 979.71515067, 986.50585105
CX, CY = 448.7607866, 354.91012286
'''
Those are for Track2
'''
# W, H = 1440, 928
# FX, FY = 1040.18078, 1038.55506
# CX, CY = 720.04463, 464.33648

'''
Those are for Track1
'''
# W, H = 640, 480
# FX, FY = 653.143433778113, 657.670567367976
# CX, CY = 299.1738577337179, 236.60674857178367


K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])

def load_pose_csv(file_path):
    """
    Robustly loads pose CSV in either Quaternion (9-col) or Matrix (17-col) format.
    Returns: (list of 4x4 matrices, list of int64 timestamps)
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None, None

    try:
        df = pd.read_csv(file_path)
        
        # Ensure timestamp is treated as 64-bit integer for precision
        possible_keys = ['timestamp', 'timestamps', 'ts', 'time']
        ts_col = next((col for col in df.columns if col.lower() in possible_keys), None)
        if ts_col is None:
            print("Warning: No timestamp key found, falling back to column 0.")
            ts_col = df.columns[0]
        timestamps = df[ts_col].astype(np.int64).values
        
        poses = []

        # Check column count to determine format
        # Format A: id, ts, x, y, z, qx, qy, qz, qw (9 columns)
        if df.shape[1] == 9:
            data = df.iloc[:, 2:].values # Extract x, y, z, qx, qy, qz, qw
            for row in data:
                T = np.eye(4)
                T[:3, 3] = row[:3] # x, y, z
                # Scipy expects [qx, qy, qz, qw]
                T[:3, :3] = R.from_quat(row[3:]).as_matrix()
                poses.append(T)

        # Format B: ts, m00...m33 (17 columns)
        elif df.shape[1] == 17:
            data = df.iloc[:, 1:].values # Skip timestamp
            for row in data:
                T = row.reshape(4, 4)
                poses.append(T)

        else:
            raise ValueError(f"Unknown CSV format with {df.shape[1]} columns.")

        return poses, timestamps

    except Exception as e:
        print(f"Failed to load Pose CSV: {e}")
        return None, None

def build_map(lidar_poses, pose_ts, pcd_dir, voxel_size=0.1, chunk_size=5):
    """
    Builds global map using a chunk-based approach to estimate 
    high-accuracy surface normals oriented towards the sensor trajectory.
    """
    
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
    if not pcd_files or len(lidar_poses) == 0:
        print("No PCD files or poses provided to build_map.")
        return o3d.geometry.PointCloud()
        
    combined_pcd = o3d.geometry.PointCloud()
    pose_ts = np.array(pose_ts, dtype=np.int64)

    chunk_pcd = o3d.geometry.PointCloud()
    chunk_origins = []

    print(f"Syncing and Merging {len(pcd_files)} PCD frames (Chunk size: {chunk_size})...")
    
    for local_idx, pcd_path in tqdm(enumerate(pcd_files), total=len(pcd_files)):
        try:
            file_ts = np.int64(os.path.splitext(os.path.basename(pcd_path))[0])
        except ValueError:
            continue

        closest_idx = np.argmin(np.abs(pose_ts - file_ts))

        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points)
        if pts.shape[1] > 3:
            pts = pts[:, :3]
            
        '''
        [Optional]:
         
        You can filter out some point for better look
        (maybe better gaussian init point guess)
        '''
        pcd.points = o3d.utility.Vector3dVector(pts)
        
        
        # Transform to global coordinates
        current_pose = lidar_poses[closest_idx]
        pcd.transform(current_pose)
        
        # Accumulate into chunk buffer
        chunk_pcd += pcd
        # Store absolute sensor position (XYZ)
        chunk_origins.append(current_pose[:3, 3])

        # --- Chunk-based Normal Estimation ---
        if (local_idx + 1) % chunk_size == 0 or (local_idx + 1) == len(pcd_files):
            
            # 1. Local downsampling to reduce redundancy and speed up computation
            chunk_pcd = chunk_pcd.voxel_down_sample(voxel_size=voxel_size)

            # 2. Compute local normals
            chunk_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
            )

            # 3. Orient normals towards the mean sensor origin of the chunk
            mean_sensor_origin = np.mean(chunk_origins, axis=0)
            chunk_pcd.orient_normals_towards_camera_location(mean_sensor_origin)

            # Merge chunk into global map
            combined_pcd += chunk_pcd

            # Reset chunk buffer
            chunk_pcd = o3d.geometry.PointCloud()
            chunk_origins = []

    # --- Global Post-processing ---
    print("Performing global downsampling and cleaning...")
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
    combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)

    return combined_pcd

def colorize_map(pcd, img_dir, img_poses):
    """
    Projects image colors using depth dilation and backface culling 
    based on pre-computed normals.
    """
    if img_poses is None or not os.path.exists(img_dir):
        print("Skipping colorization: Image data or poses not found.")
        return pcd

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    points = np.asarray(pcd.points)
    n_points = points.shape[0]

    # Use pre-computed high-precision normals from mapping stage
    if not pcd.has_normals():
        raise ValueError("Error: Point cloud missing normals.")
    
    normals = np.asarray(pcd.normals)

    color_sum = np.zeros((n_points, 3))
    color_counts = np.zeros((n_points, 1))
    min_depths = np.full(n_points, np.inf)

    print(f"Colorizing point cloud ({len(img_files)} images)...")
    pts_homo = np.hstack((points, np.ones((n_points, 1))))

    for i, img_path in tqdm(enumerate(img_files), total=len(img_files)):
        if i >= len(img_poses):
            break

        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        T_wc = img_poses[i]
        T_cw = np.linalg.inv(T_wc)
        
        # Transform points and normals to camera frame
        pts_cam = (T_cw @ pts_homo.T).T
        normals_cam = (T_cw[:3, :3] @ normals.T).T

        valid_z_mask = pts_cam[:, 2] > 0.1
        if not np.any(valid_z_mask): continue

        # --- Backface Culling ---
        view_dirs = -pts_cam[valid_z_mask, :3]
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
        valid_normals = normals_cam[valid_z_mask]
        
        # Filter points facing away from camera
        dot_products = np.sum(view_dirs * valid_normals, axis=1)
        facing_mask = dot_products > -0.2  

        # Project 3D points to 2D image plane (Assumes K, W, H are defined globally)
        pts_2d_homo = (K @ pts_cam[valid_z_mask, :3].T).T  
        u = (pts_2d_homo[:, 0] / pts_2d_homo[:, 2]).astype(int)
        v = (pts_2d_homo[:, 1] / pts_2d_homo[:, 2]).astype(int)
        z = pts_2d_homo[:, 2]

        valid_uv_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) 
        
        final_mask = facing_mask & valid_uv_mask
        
        valid_indices = np.where(valid_z_mask)[0][final_mask]
        u_valid = u[final_mask]
        v_valid = v[final_mask]
        z_valid = z[final_mask]

        # Occlusion handling using depth buffer
        depth_buffer = np.full((H, W), np.inf)
        for vv, uu, zz in zip(v_valid, u_valid, z_valid):
            if zz < depth_buffer[vv, uu]:
                depth_buffer[vv, uu] = zz

        # Dilate depth buffer to close voxel gaps
        dense_depth_buffer = minimum_filter(depth_buffer, size=3)

        # Multi-view color fusion
        for idx, vv, uu, zz in zip(valid_indices, v_valid, u_valid, z_valid):
            if zz <= dense_depth_buffer[vv, uu] + 0.8:
                if zz < min_depths[idx] * 0.90:
                    min_depths[idx] = zz
                    color_sum[idx] = img[vv, uu]
                    color_counts[idx] = 1
                elif zz <= min_depths[idx] * 1.15:
                    color_sum[idx] += img[vv, uu]
                    color_counts[idx] += 1
                    if zz < min_depths[idx]:
                        min_depths[idx] = zz

    # Compute final average colors
    final_colors = np.zeros((n_points, 3))
    colored_mask = (color_counts.flatten() > 0)
    final_colors[colored_mask] = color_sum[colored_mask] / color_counts[colored_mask]
    final_colors[~colored_mask] = [0.2, 0.2, 0.2] # Default gray

    # Track points that are successfully colored
    final_colored_mask = colored_mask.copy().flatten()

    # --- Neighbor Sphere Color Filling (0.35m radius) ---
    if np.any(colored_mask) and np.any(~colored_mask):
        print("Filling camera blind spots using 0.35m neighbor sphere...")
        
        colored_pts = points[colored_mask]
        colored_rgb = final_colors[colored_mask]
        spatial_tree = cKDTree(colored_pts)
        
        uncolored_indices = np.where(~colored_mask)[0]
        uncolored_pts = points[uncolored_indices]
        
        neighbors = spatial_tree.query_ball_point(uncolored_pts, r=0.5)
        
        for i, idx_list in enumerate(neighbors):
            if len(idx_list) > 0:
                global_idx = uncolored_indices[i]
                final_colors[global_idx] = np.mean(colored_rgb[idx_list], axis=0)
                final_colored_mask[global_idx] = True

    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    
    return pcd


def draw_camera_trajectory_and_axes(poses, step=20, scale=0.5):
    """
    poses: List of 4x4 transformation matrices.
    step: Interval at which to draw the coordinate frame (axis).
    scale: Size of the coordinate frame axes.
    """
    # Extract translation components for trajectory points
    points = [p[:3, 3] for p in poses]

    # 1. Create a red line set for the trajectory
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 2. Create coordinate frames at specified intervals
    geometries = [line_set]
    for i in range(0, len(poses), step):
        # Create a coordinate frame representing the camera pose
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        # Apply the transformation matrix to align the frame with the pose
        axis.transform(poses[i])
        geometries.append(axis)

    return geometries


def main():
    target = "Track3"

    # Lidar & Pose for mapping
    RAW_LIDAR_DIR = f"data/{target}/data/raw_pcd"
    LIDAR_POSE_PATH = f"data/{target}/result/lidar_poses.csv"

    # Image & Pose for Colorization
    IMG_DIR = f"data/{target}/data/image"
    IMG_POSE_PATH = f"data/{target}/result/camera_pose.csv"

    # Output
    OUTPUT_PCD = f"data/{target}/result/{target}.pcd"

    # ---------------------------------------------------------
    # Map Building Section]
    # 1. Load Poses and building map
    poses, timestamps = load_pose_csv(LIDAR_POSE_PATH)

    if poses is None or timestamps is None:
        print("Lack of timestamp or poses")
        return
    full_map = build_map(poses, timestamps, RAW_LIDAR_DIR, voxel_size=0.1)
    o3d.io.write_point_cloud("output_map.pcd", full_map)
    print("Map processing complete.")

    # 2. Outlier Removal
    # nb_neighbors: Number of neighbors to analyze
    # std_ratio: Threshold based on standard deviation (lower = more aggressive)
    print("Performing outlier removal...")
    full_map, ind = full_map.remove_statistical_outlier(nb_neighbors=80, std_ratio=2.0)
    # Optional: Downsample if the SLAM map is too dense
    full_map = full_map.voxel_down_sample(voxel_size=0.2)

    # 3. Image Colorization
    print("Starting colorization...")
    img_poses, _ = load_pose_csv(IMG_POSE_PATH)
    if img_poses is not None:
        full_map = colorize_map(full_map, IMG_DIR, img_poses)
    else:
        print("Warning: Failed to load camera poses. Skipping colorization.")

    # 4. Save and Visualization
    o3d.io.write_point_cloud(OUTPUT_PCD, full_map)
    print(f"Colored map saved to: {OUTPUT_PCD}")

    print("\n[INFO] Visualization Legend:")
    print("Red lines: Camera Trajectory")
    print("Axes Colors: X-axis (Red), Y-axis (Green), Z-axis (Blue)")
    print("----------------------------------------------------------\n")

    # pose and orientation
    cam_geometries = draw_camera_trajectory_and_axes(img_poses, step=20, scale=0.8)

    o3d.visualization.draw_geometries(
        [full_map] + cam_geometries,
        window_name="Refined Color Map with Camera Path",
        width=1280,
        height=720,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    main()