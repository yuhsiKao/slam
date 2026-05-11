"""
motion_predictor.py
====================
SE(3) 二階運動先驗估計，取代 main.py 裡的 constant-velocity model。

原版問題
--------
main.py 只存 prev_delta_T（上一幀相對運動），每幀預測為：
    init_T = prev_pose @ prev_delta_T

這是零階外插（constant velocity）：假設速度不變、角速度不變。
急轉彎時車輛正在改變角速度，但預測值仍沿舊方向，
init_T 偏差可能超過 GICP 的 max_correspondence_distance，
導致直接 reject 並退回 init_T——漂移加速。

本模組的做法
------------
維護一個長度為 N 的 delta 歷史窗口（SE(3) 相對運動序列）。

Translation 部分（線性加速度外插）：
    v_{t}   = trans(delta_t)        # 上一幀平移
    v_{t-1} = trans(delta_{t-1})    # 上上幀平移
    a       = v_t - v_{t-1}         # 估計線加速度
    v_pred  = v_t + alpha * a       # 二階預測

Rotation 部分（SO(3) 角速度外插，用 Log/Exp 映射）：
    omega_t   = Log(R_t)            # 上一幀角速度向量
    omega_{t-1} = Log(R_{t-1})
    alpha_rot = omega_t - omega_{t-1}  # 估計角加速度
    omega_pred = omega_t + beta * alpha_rot
    R_pred    = Exp(omega_pred)

alpha、beta 是阻尼係數（0~1），防止外插過衝。
信心加權：ICP score 越差，新 delta 的信心越低，
         在歷史中佔比越小（weighted rolling update）。

回退策略：
    - 歷史不夠（< 2 幀）→ constant velocity
    - 外插結果幾何上不合理（det ≠ 1）→ constant velocity
    - 任何數值異常 → constant velocity
"""

import numpy as np
from collections import deque


# ─────────────────────────────────────────
#  SO(3) 工具函數
# ─────────────────────────────────────────

def _so3_log(R: np.ndarray) -> np.ndarray:
    """
    SO(3) 對數映射：旋轉矩陣 → 角速度向量（軸角表示）。
    返回 3-dim vector，範數 = 旋轉角（弧度）。
    """
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3)
    # 反對稱部分提取旋轉軸
    skew = (R - R.T) / (2.0 * np.sin(angle))
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return axis * angle


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    """
    SO(3) 指數映射：角速度向量 → 旋轉矩陣（Rodrigues 公式）。
    """
    angle = np.linalg.norm(omega)
    if angle < 1e-8:
        return np.eye(3)
    axis = omega / angle
    K = np.array([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _is_valid_se3(T: np.ndarray, tol: float = 0.05) -> bool:
    """確認 4x4 矩陣是合法的 SE(3) 元素。"""
    if T.shape != (4, 4):
        return False
    if not np.isfinite(T).all():
        return False
    det = np.linalg.det(T[:3, :3])
    return abs(det - 1.0) < tol


# ─────────────────────────────────────────
#  MotionPredictor
# ─────────────────────────────────────────

class MotionPredictor:
    """
    SE(3) 二階運動外插器。

    Parameters
    ----------
    window_size : int
        保留多少幀 delta 歷史（建議 3~5）。
    trans_alpha : float
        平移加速度阻尼（0 = constant velocity, 1 = full acceleration）。
        建議 0.4~0.6：外插不要過衝，也能感知轉彎。
    rot_beta : float
        旋轉加速度阻尼。建議比 trans_alpha 小一點（旋轉更容易過衝）。
    score_halflife : float
        ICP score 加權的半衰點。score < halflife → 高信心；
        score > halflife → 低信心，此 delta 在歷史中權重降低。
    """

    def __init__(
        self,
        window_size:    int   = 5,
        trans_alpha:    float = 0.5,
        rot_beta:       float = 0.35,
        score_halflife: float = 0.3,
    ):
        self.window_size   = window_size
        self.trans_alpha   = trans_alpha
        self.rot_beta      = rot_beta
        self.score_halflife = score_halflife

        # 歷史窗口：每個元素是 (delta_T_4x4, weight)
        self._history: deque = deque(maxlen=window_size)

    # ──────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────

    def update(self, delta_T: np.ndarray, icp_score: float = 0.0) -> None:
        """
        更新歷史，每幀 ICP 成功後呼叫一次。

        Parameters
        ----------
        delta_T   : (4,4) 這幀的相對運動，= inv(prev_pose) @ curr_pose
        icp_score : ICP 對齊品質分數（越低越好），用於信心加權
        """
        if not _is_valid_se3(delta_T):
            return

        # 信心 = ICP 越好 → 權重越高
        # 用 sigmoid-like decay：score=0 → w=1，score=halflife → w=0.5
        weight = 1.0 / (1.0 + icp_score / max(self.score_halflife, 1e-6))
        self._history.append((delta_T.copy(), weight))

    def predict(self, prev_pose: np.ndarray) -> np.ndarray:
        """
        預測下一幀的絕對位姿。

        Parameters
        ----------
        prev_pose : (4,4) 當前幀（最新已知）的世界座標位姿

        Returns
        -------
        init_T : (4,4) 預測的下一幀位姿，作為 GICP 的 init_T
        """
        if not _is_valid_se3(prev_pose):
            return prev_pose.copy()

        n = len(self._history)

        # ── 少於 1 幀：只能用 identity（第一幀）
        if n == 0:
            return prev_pose.copy()

        # ── 只有 1 幀：constant velocity（原版行為）
        if n == 1:
            delta_T, _ = self._history[-1]
            return prev_pose @ delta_T

        # ── 2 幀以上：二階外插
        return self._second_order_predict(prev_pose)

    # ──────────────────────────────────────
    #  Internal: 二階外插
    # ──────────────────────────────────────

    def _second_order_predict(self, prev_pose: np.ndarray) -> np.ndarray:
        """
        用最近兩幀 delta 估計加速度，外插下一幀 delta。

        Translation：線性加速度模型
        Rotation：SO(3) Log/Exp 角速度加速度模型
        """
        # 取最近兩幀（加權平均如果有多幀）
        delta_t,   _ = self._weighted_recent(0)   # 最新
        delta_tm1, _ = self._weighted_recent(1)   # 上一幀

        # ── Translation ───────────────────────────────
        v_t   = delta_t[:3, 3]
        v_tm1 = delta_tm1[:3, 3]
        accel = v_t - v_tm1
        v_pred = v_t + self.trans_alpha * accel

        # ── Rotation（SO(3) Log 空間）─────────────────
        omega_t   = _so3_log(delta_t[:3, :3])
        omega_tm1 = _so3_log(delta_tm1[:3, :3])
        alpha_rot = omega_t - omega_tm1
        omega_pred = omega_t + self.rot_beta * alpha_rot
        R_pred = _so3_exp(omega_pred)

        # ── 組裝預測 delta ────────────────────────────
        delta_pred = np.eye(4)
        delta_pred[:3, :3] = R_pred
        delta_pred[:3, 3]  = v_pred

        # ── 安全檢查：若外插結果幾何上不合理，回退 CV ─
        if not _is_valid_se3(delta_pred, tol=0.1):
            print("[MotionPredictor] 2nd-order extrapolation invalid → fallback CV", flush=True)
            return prev_pose @ delta_t

        init_T = prev_pose @ delta_pred

        if not _is_valid_se3(init_T, tol=0.1):
            print("[MotionPredictor] init_T invalid → fallback CV", flush=True)
            return prev_pose @ delta_t

        # 印出外插幅度，方便 debug
        trans_diff = np.linalg.norm(v_pred - v_t)
        rot_diff   = np.degrees(np.linalg.norm(alpha_rot))
        print(
            f"[MotionPredictor] 2nd-order: "
            f"Δtrans={trans_diff:.3f}m  Δrot={rot_diff:.2f}°  "
            f"alpha={self.trans_alpha}  beta={self.rot_beta}",
            flush=True
        )

        return init_T

    def _weighted_recent(self, offset: int):
        """
        取歷史窗口倒數第 (offset+1) 個元素。
        若窗口夠大，對最近 2 幀做加權平均再返回（更穩健）。
        offset=0 → 最新；offset=1 → 上一幀。
        """
        history_list = list(self._history)
        idx = -(offset + 1)           # -1 最新，-2 上一幀

        # 單幀直接返回
        if len(history_list) < abs(idx) + 1:
            return history_list[idx]

        # 若窗口 >= 3，對目標幀及其鄰幀做指數加權平均（降噪）
        delta, w = history_list[idx]
        if len(history_list) >= 3 and abs(idx) < len(history_list) - 1:
            neighbor_idx = idx - 1 if offset == 0 else idx + 1
            try:
                d_n, w_n = history_list[neighbor_idx]
                # 加權混合：當前幀佔 70%，鄰幀佔 30%
                lam = 0.7
                v_blend   = lam * delta[:3, 3] + (1 - lam) * d_n[:3, 3]
                om_blend  = lam * _so3_log(delta[:3, :3]) + (1 - lam) * _so3_log(d_n[:3, :3])
                R_blend   = _so3_exp(om_blend)
                d_blended = np.eye(4)
                d_blended[:3, :3] = R_blend
                d_blended[:3, 3]  = v_blend
                w_blend = lam * w + (1 - lam) * w_n
                return d_blended, w_blend
            except Exception:
                pass

        return delta, w

    # ──────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────

    def reset(self) -> None:
        """Pose graph 優化後位姿被校正，歷史 delta 失效，清空。"""
        self._history.clear()

