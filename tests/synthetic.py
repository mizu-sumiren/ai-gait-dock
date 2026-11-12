"""答えの分かっている人工の歩行データを作る（テスト用）。"""
import numpy as np
from gait.pose import (PoseSeries, to_pixels, N_LANDMARKS, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
                       L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)


def side_walker(fps=60.0, stride_s=1.1, lean_deg=6.0, knee_deg=8.0, width=1920, height=1080,
                near="right", direction=1, duration=4.5, drop_frames=()):
    n = int(duration * fps)
    t = np.arange(n) / fps
    L, Lt = 180.0, 220.0                      # 大腿・下腿の長さ、体幹の長さ（ピクセル）
    xy = np.full((n, N_LANDMARKS, 2), np.nan)
    vis = np.zeros((n, N_LANDMARKS))
    speed = 1.3 * 2 * L * np.sin(np.radians(25)) / (stride_s / 2) / 1.3
    x0 = 200 if direction > 0 else width - 200
    hip = np.stack([x0 + direction * speed * t, np.full(n, 520.0)], axis=1)
    lean = np.radians(lean_deg)
    sh = hip + Lt * np.array([np.sin(lean) * direction, -np.cos(lean)])
    for side, hip_i, knee_i, ank_i, sh_i, ph in (("left", L_HIP, L_KNEE, L_ANKLE, L_SHOULDER, 0.0),
                                                 ("right", R_HIP, R_KNEE, R_ANKLE, R_SHOULDER, np.pi)):
        phi = (2 * np.pi * t / stride_s + ph) % (2 * np.pi)
        th = np.radians(25) * np.cos(phi)
        k = np.radians(knee_deg) + np.radians(55) * np.where(phi > np.pi, np.sin(phi - np.pi) ** 2, 0.0)
        knee = hip + L * np.stack([np.sin(th) * direction, np.cos(th)], axis=1)
        ank = knee + L * np.stack([np.sin(th - k) * direction, np.cos(th - k)], axis=1)
        xy[:, hip_i], xy[:, knee_i], xy[:, ank_i], xy[:, sh_i] = hip, knee, ank, sh
        v = 0.95 if side == near else 0.6
        vis[:, [hip_i, knee_i, ank_i, sh_i]] = v
    for f in drop_frames:                     # 未検出フレーム
        xy[f] = np.nan
        vis[f] = 0
    return PoseSeries(t=t, xy=xy, vis=vis, width=width, height=height, fps=fps)


def front_walker(fps=60.0, stride_s=1.1, cv_pct=3.0, n_strides=24, seed=0, approach=True):
    rng = np.random.default_rng(seed)
    durs = stride_s * (1 + cv_pct / 100 * rng.standard_normal(n_strides))
    edges = np.concatenate([[0], np.cumsum(durs)])
    n = int(edges[-1] * fps)
    t = np.arange(n) / fps
    phase = np.interp(t, edges, 2 * np.pi * np.arange(n_strides + 1))
    scale = np.linspace(0.5, 1.5, n) if approach else np.ones(n)   # 近づくほど大きく映る
    xy = np.full((n, N_LANDMARKS, 2), np.nan)
    vis = np.full((n, N_LANDMARKS), 0.9)
    cx, top = 540.0, 600.0
    torso = 200.0 * scale
    for i, sx in ((L_SHOULDER, 0.45), (R_SHOULDER, -0.45), (L_HIP, 0.3), (R_HIP, -0.3)):
        y = top if i in (L_SHOULDER, R_SHOULDER) else top + torso
        xy[:, i] = np.stack([cx + sx * torso, np.broadcast_to(y, n)], axis=1)
    d = 0.12 * torso * np.sin(phase)
    for i, sgn, sx in ((L_ANKLE, 1, 0.2), (R_ANKLE, -1, -0.2)):
        xy[:, i] = np.stack([cx + sx * torso, top + 3.2 * torso + sgn * d / 2], axis=1)
    xy[:, [L_KNEE, R_KNEE]] = xy[:, [L_HIP, R_HIP]]
    return PoseSeries(t=t, xy=xy, vis=vis, width=1080, height=1920, fps=fps), durs
