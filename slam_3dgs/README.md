# 2026 Spring SDC Midterm Competition II - SLAM with 3DGS

## Code Structure
```
3dgs_slam
├── generate_sky_mask.py
├── output/                  # To put the results
├── scene
│   └── dataset_readers.py      # Load the dataset
├── test_camera_poses.py        # To check if the camera poses match the map
├── train.py                 # Main trainer
├── utils
│   └── general_utils.py
├── test_frame_list.txt      # Test frame list (timestamps)
├── submission.csv           # Put into your submission file
├── generate_test_pose.py    # Estimate the test poses
└── generate_submission.py   # Render test images
```

## Dataset Format
```
itri58_colored_pcd
├── camera_gt_pose.txt
├── camera_intrinsics.json
├── itri58_full_color_map.pcd
├── itri58_image
│   ├── 1741574911912996024.jpg
│   └── ...
├── read_pcd._example.py       # Visualize the pointcloud
└── sky_masks/                 # To be created
```

## Environment Setup (Tested on RTX 5090, Ubuntu 22.04)
```bash
# 1. Create a fresh environment
conda create -n 3dgs_new python=3.11 -y
conda activate 3dgs_new

# 2. Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install gsplat and other 3DGS essentials
pip install gsplat 

# 4. Install remaining dependencies in one go
pip install opencv-python open3d scipy tensorboard numpy tqdm

# 5. Install transformers for DinoV2 sky masking
pip install transformers timm pillow xformers
```

## Training Steps
1. Run `test_camera_poses.py` and get `visualization.ply`, you will see the camera traj and the map, make sure they match each other
2. Run `generate_sky_mask.py` to generate sky masks
3. Run `train_v2.py` to train

## Evaluation Steps
1. Run `generate_test_pose.py` to estimate test poses
2. Run `generate_submission.py` to render test images
3. Compress the 30 test images and `submission.csv` into `submission.zip` (or other name you like)
4. Upload `submission.zip` to kaggle