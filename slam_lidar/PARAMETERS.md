# SLAM LiDAR — 參數文件

> 最後更新：2026-05-10  
> 場景：Track3（室內窄走廊 + 室外長直線混合）

---

## 1. 環境偵測 (`ICP.py`)

| 參數 | 值 | 說明 |
|------|-----|------|
| `_INDOOR_RANGE_P95` | `40.0 m` | p95 點雲距離低於此值視為 indoor |
| `_vote_window_in` | `1` | 連續幾幀 indoor vote → 切換到 indoor（快切） |
| `_vote_window_out` | `3` | 連續幾幀 outdoor vote → 切回 outdoor（慢切，防誤報） |

> **調整提示**：看 log 中 `p95=XX.Xm` 的分布，找 indoor/outdoor 的自然分界點來設定 `_INDOOR_RANGE_P95`。

---

## 2. ICP 參數 (`ICP.py`)

### 2.1 Per-environment 設定

| 參數 | Indoor | Outdoor | 說明 |
|------|--------|---------|------|
| `voxel_size` | `0.20 m` | `0.30 m` | GICP 降採樣解析度 |
| `max_correspondence_distance` | `1.5 m` | `2.0 m` | 對應點最大搜尋距離（建議 3–5× voxel） |
| `max_range` | `30.0 m` | `100.0 m` | source 點雲最大有效距離（sensor frame） |
| `max_iterations` | `50` | `80` | GICP 最大迭代數（early stopping 通常提前結束） |
| `registration_type` | `GICP` | `GICP` | 配準方法 |
| `min_z` (source) | `-2.0 m` | `-5.0 m` | source sensor frame Z 下限 |
| `max_z` (source) | `10.0 m` | `30.0 m` | source sensor frame Z 上限 |
| `min_z` (target) | `-5.0 m` | `-5.0 m` | target world frame Z 下限（`self.min_z`，固定） |
| `max_z` (target) | `30.0 m` | `30.0 m` | target world frame Z 上限（`self.max_z`，固定） |

> **重要**：target（world-frame submap）不做 `max_range` 過濾（使用 `1e9`），避免機器人走遠後 submap 被全部過濾掉。

### 2.2 全域設定

| 參數 | 值 | 說明 |
|------|-----|------|
| `num_threads` | `4` | 平行運算執行緒數 |
| `num_neighbors` | `20` | 法向量估算用的鄰近點數 |
| `reject_score` | `1.0` | ICP error/n_ds 超過此值 → reject（目前 GICP error ≈ 0，此門檻實際無效） |
| `max_delta_trans` | `5.0 m` | ICP 結果相對 init\_T 的平移上限，超過 → reject |
| `max_delta_rot` | `30.0°` | ICP 結果相對 init\_T 的旋轉上限，超過 → reject |
| `min_range` | `0.5 m` | 過濾過近點（sensor 自身遮擋） |
| `min_points` | `50` | source/target 少於此點數 → 跳過 ICP |
| `early_stop_delta` | `1e-4 m` | 收斂閾值：平移步長 |
| `early_stop_rot` | `1e-4 rad` | 收斂閾值：旋轉步長（≈ 0.006°） |
| `early_stop_chunk` | `5` | 每幾個 iterations 檢查一次收斂 |

---

## 3. Submap 管理 (`submap.py`)

| 參數 | 值 | 說明 |
|------|-----|------|
| `window_size` | `3` | submap 前後各涵蓋幾個 keyframe（共 ≤ 2w+1 幀） |
| `submap_voxel` | `0.05 m` | submap 合併後的 voxel 降採樣解析度 |

> `submap_window = 5`（轉彎偵測時）：`turn_angle_deg > 2.5°` 時使用較寬的 submap window。

---

## 4. Keyframe 管理 (`keyframe.py`)

| 參數 | 值 | 說明 |
|------|-----|------|
| `dist_thresh` | `1.0 m` | 距上一個 keyframe 超過此距離 → 加新 keyframe |
| `ang_thresh` | `10°` | 旋轉超過此角度 → 加新 keyframe |

---

## 5. 運動預測器 (`motion_predictor.py`)

| 參數 | main.py 設定 | 預設值 | 說明 |
|------|------------|--------|------|
| `window_size` | `4` | `4` | 保留幾幀 delta 歷史 |
| `trans_alpha` | `0.5` | `0.3` | 平移加速度阻尼（0=常速，1=全加速） |
| `rot_beta` | `0.35` | `0.35` | 旋轉加速度阻尼（比 trans_alpha 小以防過衝） |
| `score_halflife` | `0.3` | `0.3` | ICP score 信心加權半衰點 |

> 轉角發散時可降低 `rot_beta`（試 0.2）；直線偏移時可降低 `trans_alpha`（試 0.3）。

---

## 6. Loop Closure (`loopclosure.py`)

### 6.1 FPFH 描述子參數

| 參數 | Indoor | Outdoor | 說明 |
|------|--------|---------|------|
| `voxel` | `0.15 m` | `0.50 m` | FPFH 計算前降採樣 |
| `normal_r` | `0.5 m` | `1.5 m` | 法向量估算半徑 |
| `feature_r` | `1.0 m` | `3.0 m` | FPFH 特徵半徑 |
| `ransac_iters` | `2000` | `4000` | RANSAC 最大迭代數 |

| 全域 FPFH 參數 | 值 | 說明 |
|--------------|-----|------|
| `_DESC_DIST_THRESH` | `60.0` | 描述子 L2 距離超過此值 → 排除候選 |
| `_RANSAC_DIST` | `1.0 m` | RANSAC 對應點容忍距離 |
| `_RANSAC_CONF` | `0.95` | RANSAC 置信度 |
| `_FPFH_CACHE_MAX` | `150` | 最多快取幾幀的 FPFH 特徵 |

### 6.2 Loop Closure 搜尋參數

| 參數 | 值 | 說明 |
|------|-----|------|
| `dist_thresh` | `8.0 m` | 空間 KD-tree 搜尋半徑（outdoor 自動 ×1.5 = 12m） |
| `fitness_thresh` | `0.25` | ICP score 超過此值 → reject loop |
| `max_candidates` | `20` | 最多同時評估幾個候選 |
| `min_separation` | `15` | 候選 keyframe 必須與當前幀至少相差幾幀 |
| `max_loop_correction` | `5.0 m` | loop 修正量上限（outdoor ×2.5 = 12.5m） |
| `confirm_thresh` | `2` | 連續幾次偵測到同一 loop → 才接受 |
| `desc_top_k` | `3` | 描述子排序後保留幾個候選進 RANSAC+ICP |

---

## 7. Pose Graph (`posegraph.py`)

| 參數 | 值 | 說明 |
|------|-----|------|
| Prior noise sigma | `1e-3` | 起始位置約束（極緊，固定世界座標原點） |
| Odometry edge sigma | `0.05 ~ 0.30` | 動態計算：`clip(0.05 + icp_delta × 0.08, 0.05, 0.30)` |
| Loop closure base sigma | `0.05` | 比 odometry 緊，搭配 Cauchy robust kernel |
| Cauchy kernel scale | `1.0` | 超過此 scale 的殘差使用 M-estimator 降權 |

---

## 8. 主迴圈 (`main.py`)

| 參數 | 值 | 說明 |
|------|-----|------|
| `optimize_every` | `10` | 每累積幾個 keyframe 執行一次 pose graph 優化（≈ 每 10m） |
| Submap turn threshold | `2.5°` | prev_delta_T 旋轉角超過此值 → 使用 `window=5` 寬 submap |
| Visual map voxel | `0.5 m` | 全域視覺化地圖的降採樣解析度（不影響 ICP） |
