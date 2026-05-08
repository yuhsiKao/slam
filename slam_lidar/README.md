# 2026 Spring SDC Midterm Competition II - SLAM with 3DGS

## Code Structure
```
.
├── main.py 
├── process            # Processing pose results into a colored map for GS
│   ├── camera_pose.py
│   ├── color_mapping.py
│   └── read_pcd_example.py
└── utils
    ├── ICP.py         # TODO 1: scan-matching ALGO
    ├── io.py
    ├── keyframe.py
    ├── loopclosure.py # TODO 2: loop-closure detection and tf estimation
    ├── posegraph.py
    ├── submap.py
    └── viz.py
```


## Dataset Format
```
Track1
├── data
│   ├── image/
│   └── raw_pcd/
└── result
    ├── camera_pose.csv
    └── lidar_poses.csv
```

## Environment Setup
```
# Create a fresh environment
conda create -n slam_env python=3.9
conda activate slam_env

# Core numerical computing and progress bars
conda install numpy pandas scipy tqdm

pip install opencv-python
pip install open3d

# Factor graph optimization
pip install gtsam
pip install small_gicp
```

## Build map and color the lidar points

### Make sure you set the correct paths in the script before running.
1. Finish the TODOs in `/utils/ICP.py` & `/utils/ICP.py`
2. Run `main.py` to estimate the poses for each LiDAR input. 
Submit the result `lidar_poses.csv` to Kaggle.
https://www.youtube.com/watch?v=L8-utFkSwM4

#### The following steps describe how to prepare a colored PCD map for 3DGS:
3. Run `camera_pose.py` under /process to derive all camera poses from LiDAR poses.
4. Run `color_mapping.py` to color PCD the corresponding pcd result will be saved under TrackX/result/, if you want to observe PCD result again, `run read_pcd_example.py`.
![ScreenCapture_2026-04-29-20-58-27](https://hackmd.io/_uploads/HkogWzlRbx.png)
