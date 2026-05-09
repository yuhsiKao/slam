# 2026 Spring SDC Midterm Competition II — SLAM + 3DGS 系統架構文件

---

## 目錄

[TOC]

---

## 專案概覽

本專案分為兩個模組：

| 模組 | 功能 |
|------|------|
| `slam_lidar` | LiDAR SLAM：點雲掃描配準、位姿估計、地圖建構、彩色化 |
| `slam_3dgs` | 3D Gaussian Splatting：場景訓練、Novel View Synthesis、Kaggle 提交 |

整體流程：**LiDAR SLAM → 彩色點雲地圖 → 3DGS 訓練 → 測試幀渲染 → 提交**

---

## 一、slam_lidar

### 1.1 目錄結構

```
slam_lidar/
├── main.py                        # 主 SLAM Pipeline
├── convert_slam_to_3dgs.py        # 將 SLAM 輸出轉換為 3DGS 輸入格式
├── process/
│   ├── camera_pose.py             # 從 LiDAR 位姿推算相機位姿
│   ├── color_mapping.py           # 點雲彩色化（基本版）
│   └── color_mapping_norm.py      # 點雲彩色化（法向量 + Backface Culling 版）
├── utils/
│   ├── ICP.py                     # GICP 掃描配準
│   ├── io.py                      # PCD 讀取工具
│   ├── keyframe.py                # 關鍵幀管理
│   ├── loopclosure.py             # 迴環偵測
│   ├── posegraph.py               # 位姿圖最佳化（GTSAM）
│   ├── submap.py                  # 子地圖管理
│   └── viz.py                     # Open3D 即時視覺化
└── itri58_colored_pcd_t1/         # Track1 的 3DGS 輸入資料集
    ├── camera_gt_pose.txt         # 相機位姿（每行 16 個 float，逗號分隔）
    ├── camera_intrinsics.json     # 相機內參
    ├── itri58_full_color_map.pcd  # 彩色點雲地圖（symlink）
    ├── itri58_image/              # 所有訓練影像
    └── sky_masks/                 # 天空遮罩（由 3DGS 端產生）
```

---

### 1.2 主 Pipeline：`main.py`

```
原始 PCD 序列
     │
     ▼
[Frame 0] 初始化位姿 = I
     │
     ▼ (每幀)
[ICP.align] scan-to-submap 配準
  ├─ 偵測 Indoor/Outdoor 環境（自適應參數）
  ├─ 用上一幀位移作為 constant-velocity 初值
  └─ 輸出 curr_pose + ICP score
     │
     ▼
[KeyframeManager] 判斷是否為關鍵幀
  └─ 條件：位移 > 1m 或 旋轉 > 10°
     │
     ▼ (是關鍵幀)
[SubmapManager] 建立子地圖（滑動視窗合併鄰近關鍵幀）
[LoopClosure] 搜尋迴環候選幀
  ├─ KD-Tree 搜尋位置相近的歷史幀
  ├─ ICP 確認幾何吻合度
  └─ 連續確認 N 次才接受（防止誤判）
     │
     ▼ (偵測到迴環 or 每 20 幀)
[PoseGraph.optimize] GTSAM iSAM2 增量位姿最佳化
  ├─ 里程計邊（BetweenFactor, σ=0.1）
  └─ 迴環邊（BetweenFactor + Cauchy Robust Kernel, σ=0.05）
     │
     ▼
[distribute_pose_corrections] 用 Slerp 將最佳化修正量插值回所有非關鍵幀
     │
     ▼
存為 lidar_poses.csv（timestamp + 4×4 矩陣攤平）
```

---

### 1.3 ICP 模組：`utils/ICP.py`

使用 `small_gicp` 函式庫實作 GICP（Generalized ICP）。

**環境自適應機制：**

| 參數 | Indoor | Outdoor |
|------|--------|---------|
| Voxel Size | 0.10 m | 0.25 m |
| Max Correspondence Distance | 1.5 m | 4.0 m |
| Max Range | 30 m | 100 m |

- 以點雲距離的 P95 值判斷室內/室外，加入 Hysteresis 避免抖動
- 連續 5 幀同向投票才切換環境模式

**拒絕機制：**
- `delta_trans > 5.0m` → 拒絕（位移過大）
- `score > reject_score (1.0)` → 拒絕（配準品質差）

---

### 1.4 Loop Closure：`utils/loopclosure.py`

```
當前關鍵幀位置
     │
     ▼
KD-Tree 搜尋半徑內歷史幀（outdoor: 12m, indoor: 8m）
過濾時間上太近的幀（min_separation = 15 幀）
     │
     ▼
對每個候選幀呼叫 ICP.align
     ├─ 計算位移修正量 correction
     ├─ correction > limit → 拒絕
     └─ score >= fitness_thresh → 拒絕
     │
     ▼
選出最佳候選幀，須連續 confirm_thresh=2 次才確認
     │
     ▼
計算 T_loop = inv(pose_candidate) @ T_aligned
加入 PoseGraph 作為迴環邊
```

---

### 1.5 相機位姿推算：`process/camera_pose.py`

從 LiDAR 位姿插值出相機位姿：

```
T_camera_world = T_lidar_world(插值) @ T_lidar_to_camera(Extrinsic)
```

**Track1 Extrinsic（LiDAR → Camera）：**
- 平移：(-0.190, -0.115, 0.070)
- 旋轉：四元數 (-0.7071, 0, 0, 0.7071)
- 額外 Pitch：+6.5°，額外 Yaw：-3.25°

**插值方式：**
- 位移：線性插值
- 旋轉：Slerp（球形線性插值）

輸出：`camera_pose.csv`（timestamp + 4×4 矩陣，17 欄）

---

### 1.6 點雲彩色化

#### 基本版：`color_mapping.py`
- Z-buffer 遮擋處理
- 多幀顏色累加取平均

#### 增強版：`color_mapping_norm.py`
- **Chunk-based 法向量估計**：每 5 幀一組，以感測器中心定向法向量
- **Backface Culling**：過濾法向量朝向相機背面的點（dot product > -0.2）
- **Depth Dilation**：用 3×3 minimum filter 填補深度空洞
- **Multi-view 顏色融合**：
  - 深度明顯更近（< 90%）→ 重置顏色
  - 深度相近（< 115%）→ 累加平均
- **Neighbor Sphere Filling（0.5m）**：對無法被相機覆蓋的點，用周圍 0.5m 內有色點的平均顏色填補

---

### 1.7 資料格式轉換：`convert_slam_to_3dgs.py`

將 SLAM 輸出整理為 3DGS 所需格式：

```
data/Track1/
├── data/image/*.jpg          →  itri58_colored_pcd_t1/itri58_image/
├── result/camera_pose.csv    →  camera_gt_pose.txt（逗號分隔 16 floats/行）
└── result/Track1.pcd         →  itri58_full_color_map.pcd
```

同時生成 `camera_intrinsics.json`（各 Track 的 fx/fy/cx/cy/width/height）。

---

### 1.8 Track 相機內參

| Track | Width | Height | fx | fy | cx | cy |
|-------|-------|--------|-----|-----|-----|-----|
| Track1 | 640 | 480 | 653.14 | 657.67 | 299.17 | 236.61 |
| Track2 | 1440 | 928 | 1040.18 | 1038.56 | 720.04 | 464.34 |
| Track3 | 960 | 720 | 979.72 | 986.51 | 448.76 | 354.91 |

---

## 二、slam_3dgs

### 2.1 目錄結構

```
slam_3dgs/
├── pipeline.py                    # 自動化全流程腳本
├── train.py                       # 3DGS 訓練器（主版本）
├── train_v2.py                    # 3DGS 訓練器（進階版，若存在）
├── generate_sky_mask.py           # 天空語義分割遮罩生成
├── generate_test_pose.py          # 測試幀位姿插值
├── generate_submission.py         # 渲染測試幀並產生提交檔
├── test_camera_poses.py           # 驗證相機位姿與地圖對齊
├── scene/
│   ├── __init__.py
│   └── dataset_readers.py         # 資料集讀取（影像、位姿、點雲、遮罩、深度）
├── utils/
│   ├── __init__.py
│   ├── general_utils.py           # 工具函式（mkdir、inverse_sigmoid 等）
│   └── sh_utils.py                # Spherical Harmonics 轉換（RGB ↔ SH DC）
├── output/                        # 訓練輸出（.ply 檔、TensorBoard logs）
├── track1/
│   ├── track1_test_frame_list.txt # 30 個測試幀時間戳
│   ├── test_pose_list.txt         # 插值後的測試位姿（由 generate_test_pose.py 產生）
│   ├── test_submission/           # 渲染輸出的 PNG 影像
│   └── submission.csv             # Kaggle 提交格式
├── track2/                        # 同上，Track2
└── track3/                        # 同上，Track3
```

---

### 2.2 完整 Pipeline：`pipeline.py`

```
對每個 Track 依序執行：

Step 1: test_camera_poses.py
  └─ 驗證相機位姿 + 點雲對齊，輸出 visualization.ply

Step 2: generate_sky_mask.py
  └─ 對所有訓練影像生成天空遮罩（.png，255=天空，0=非天空）

Step 3: train.py / train_v2.py
  └─ 以點雲初始化 Gaussian，訓練 3DGS 模型
     輸出 gaussian_reconstruction.ply

Step 4: generate_test_pose.py
  └─ 對 30 個測試時間戳插值相機位姿
     輸出 test_pose_list.txt

Step 5: generate_submission.py
  └─ 從訓練好的 .ply 渲染 30 張測試影像
     輸出 test_submission/*.png

Step 6: Zip
  └─ 將 30 張 PNG + submission.csv 打包為 submission.zip
```

執行範例：
```bash
python pipeline.py                     # 跑全部 3 個 Track
python pipeline.py --tracks 1 2        # 只跑 Track1, Track2
python pipeline.py --tracks 1 --skip_train  # 跳過訓練，直接推論
```

---

### 2.3 資料集讀取：`scene/dataset_readers.py`

`Dataset` 類別負責載入所有訓練資料：

| 方法 | 說明 |
|------|------|
| `_load_poses()` | 讀取 `camera_gt_pose.txt`，每行 16 floats → 4×4 矩陣 |
| `_load_intrinsics()` | 讀取 `camera_intrinsics.json` |
| `_load_image_list()` | 列舉 `itri58_image/` 下的影像 |
| `_load_pointcloud()` | 讀取 `itri58_full_color_map.pcd` |
| `_load_mask_list()` | 列舉 `sky_masks/` 下的遮罩 |
| `get_lidar_depth()` | 將點雲投影到指定相機幀，產生稀疏深度圖（Z-buffer） |

---

### 2.4 天空遮罩生成：`generate_sky_mask.py`

使用 HuggingFace 預訓練語義分割模型：
- **模型**：`nvidia/segformer-b5-finetuned-ade-640-640`
- **天空 Label**：ADE20K label index 2 = Sky
- **輸出**：灰階 PNG，255 = 天空，0 = 非天空

---

### 2.5 3DGS 訓練器：`train.py`

#### Gaussian 初始化

```
LiDAR 點雲（N_pc 個點）
  Scale = 0.15m（可調）
  Opacity = 0.7（logit 空間）
  Color = RGB2SH(點雲顏色)

天空點（N_sky 個點，約 10,000）
  由相機位姿 + 天空遮罩 + 隨機深度（50-80m）反投影
  Scale = 0.5m

合併 → nn.ParameterDict（means, scales, quats, opacities, sh0）
```

#### 優化器設定

| 參數 | 學習率 |
|------|--------|
| means（位置） | 1.6e-4 |
| scales（大小） | 5e-3 |
| quats（旋轉） | 1e-3 |
| opacities（不透明度） | 5e-2 |
| sh0（顏色） | 2.5e-3 |

學習率衰減：`ExponentialLR(gamma=0.9999)`

#### 渲染流程

```
camtoworld → 計算 viewmat = inv(camtoworld)
means + quats + scales(exp) + opacities(sigmoid) + colors(SH→RGB)
     │
     ▼
同時渲染 RGB（3ch）+ Depth（1ch）= 4ch features
     │
     ▼
gsplat.rasterization(near=0.01, far=1000, packed=False)
     │
     ▼
輸出 rendered_rgb [H,W,3] + rendered_depth [H,W]
```

#### 損失函數

```
L = L1(masked_render, masked_target)
```
- 天空區域（mask=255）不計入損失
- 可擴充加入 depth loss

#### Densification 策略

使用 `gsplat.DefaultStrategy`：
- **Start**：第 500 iter
- **Stop**：第 15,000 iter
- **頻率**：每 100 iter
- **Reset Opacity**：每 3,000 iter

---

### 2.6 PLY 格式規格

訓練輸出的 `.ply` 採用 Binary Little Endian，每個點 14 個 float：

```
[x, y, z]          ← 位置 (3)
[scale_0~2]        ← 大小，log 空間 (3)
[rot_0~3]          ← 四元數 (4)
[opacity]          ← logit 空間 (1)
[f_dc_0~2]         ← SH DC 顏色分量 (3)
```

---

### 2.7 測試位姿插值：`generate_test_pose.py`

```
讀取所有訓練影像的時間戳 + camera_gt_pose.txt
     │
     ▼
時間戳正規化（避免精度損失）
     │
     ▼
位移：scipy.interpolate.interp1d（linear）
旋轉：scipy.spatial.transform.Slerp
     │
     ▼
對 track{N}_test_frame_list.txt 中 30 個時間戳插值
     │
     ▼
輸出 test_pose_list.txt（每行 16 floats，空白分隔）
```

---

### 2.8 提交渲染：`generate_submission.py`

```
載入 gaussian_reconstruction.ply（StandaloneRenderer）
     │
     ▼
對每個測試幀：
  camtoworld → viewmat
  rasterization（gsplat）
  features[0, ..., :3] → RGB 影像
  composite 黑色背景
     │
     ▼
輸出 test_submission/{timestamp}.png（uint8 BGR）
```

---

### 2.9 Spherical Harmonics 工具：`utils/sh_utils.py`

```python
C0 = 0.28209479177387814   # SH DC basis coefficient

RGB2SH(rgb) = (rgb - 0.5) / C0   # 訓練時初始化 Gaussian 顏色
SH2RGB(sh)  = sh * C0 + 0.5      # 渲染時還原 RGB
```

---

## 三、資料流總覽

```
┌─────────────────────────────────────────────────────────────┐
│                        slam_lidar                           │
│                                                             │
│  raw_pcd/*.pcd  ──→  main.py  ──→  lidar_poses.csv         │
│                         │                                   │
│                   camera_pose.py                            │
│                         │                                   │
│                   camera_pose.csv                           │
│                         │                                   │
│               color_mapping_norm.py                         │
│                         │                                   │
│                   Track1.pcd（彩色）                        │
│                         │                                   │
│               convert_slam_to_3dgs.py                       │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   itri58_colored_pcd_t1/                    │
│   camera_gt_pose.txt                                        │
│   camera_intrinsics.json                                    │
│   itri58_full_color_map.pcd                                 │
│   itri58_image/*.jpg                                        │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                        slam_3dgs                            │
│                                                             │
│  generate_sky_mask.py  →  sky_masks/*.png                   │
│  train.py              →  output/gaussian_reconstruction.ply│
│  generate_test_pose.py →  track1/test_pose_list.txt         │
│  generate_submission.py→  track1/test_submission/*.png      │
│                         →  submission.zip（上傳 Kaggle）    │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、環境安裝

### slam_lidar

```bash
conda create -n slam_env python=3.9
conda activate slam_env
conda install numpy pandas scipy tqdm
pip install opencv-python open3d gtsam small_gicp
```

### slam_3dgs

```bash
conda create -n 3dgs_new python=3.11 -y
conda activate 3dgs_new
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install gsplat
pip install opencv-python open3d scipy tensorboard numpy tqdm
pip install transformers timm pillow xformers
```

---

## 五、各模組負責人 / TODO

| 模組 | 狀態 | 備註 |
|------|------|------|
| `utils/ICP.py` | ✅ 完成 | GICP + 環境自適應 |
| `utils/loopclosure.py` | ✅ 完成 | KD-Tree + 確認機制 |
| `process/color_mapping_norm.py` | ✅ 完成 | 法向量 + Backface Culling |
| `train.py` | ✅ 完成 | L1 loss + DefaultStrategy |
| 深度損失（Depth Loss） | 🔧 可擴充 | `get_lidar_depth()` 已備妥 |
| `train_v2.py` | ❓ 若存在 | pipeline 會優先使用 |