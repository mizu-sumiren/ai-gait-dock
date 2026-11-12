"""骨格データ(PoseSeries)から歩行の指標を計算する。画面(Streamlit)には依存しない。

出す指標
- 共通 : 歩行率、ストライド時間、ストライド時間のばらつき(CV)
- 側面 : 体幹の前傾角、立脚中期の膝屈曲角（どちらも参考値）

1歩ごとのタイミングは「左右の足首の離れ具合」の山から求める。
- 側面: 足首の前後の距離が最大になる時点（両脚支持期のあたり）
- 正面: 足首の上下の差が最大になる時点
左右の区別を使わないので、MediaPipeが左右を取り違えても影響を受けにくい。
"""
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from .pose import (PoseSeries, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
                   L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)

MAX_GAP_S = 0.2        # これより長い未検出区間は補間しない
MIN_BOUT_S = 1.5       # これより短い区間は解析しない
LOWPASS_HZ = 6.0
MIN_VIS = 0.5


@dataclass
class GaitResult:
    view: str
    fps: float
    detection_rate: float
    n_steps: int = 0
    n_strides: int = 0
    n_rejected_steps: int = 0
    cadence_spm: float = float("nan")          # 歩/分
    stride_time_s: float = float("nan")
    stride_time_sd_s: float = float("nan")
    stride_time_cv_pct: float = float("nan")
    trunk_lean_deg: float = float("nan")        # 前傾が＋（側面のみ）
    midstance_knee_flexion_deg: float = float("nan")   # 0°=まっすぐ（側面のみ）
    n_midstance: int = 0
    near_side: str = ""
    speed_m_s: float = float("nan")             # 手計測の距離÷時間
    stride_length_m: float = float("nan")       # 速度×ストライド時間
    messages: list = field(default_factory=list)

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("messages")
        return d


# ---------- 信号処理の小道具 ----------

def interior_angle(a, b, c):
    """点bを頂点とする角度（度）。a,b,c は (...,2)。"""
    ba, bc = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    cos = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def resample_uniform(t, y, fs, max_gap_s=MAX_GAP_S):
    """等間隔の時間軸に並べ直す。短い欠損は線形補間、長い欠損は NaN のまま。"""
    tu = np.arange(t[0], t[-1] + 0.5 / fs, 1.0 / fs)
    ok = np.isfinite(y)
    yu = np.full(len(tu), np.nan)
    if ok.sum() < 2:
        return tu, yu
    tv = t[ok]
    yu = np.interp(tu, tv, y[ok])
    idx = np.clip(np.searchsorted(tv, tu), 1, len(tv) - 1)
    gap = tv[idx] - tv[idx - 1]
    yu[(gap > max_gap_s) | (tu < tv[0]) | (tu > tv[-1])] = np.nan
    return tu, yu


def runs(mask, min_len):
    """True が続く区間 (start, end) の一覧。"""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def lowpass(y, fs, cutoff=LOWPASS_HZ):
    """ゼロ位相のローパス。filtfilt は端を折り返して処理するので端の値が落ちない。"""
    if cutoff >= 0.45 * fs or len(y) < 15:
        return np.asarray(y, dtype=float)
    b, a = butter(2, cutoff / (0.5 * fs))
    return filtfilt(b, a, y)


def _subframe(s, p):
    """山の位置を放物線で補正（フレーム間隔より細かく時刻を出す）。"""
    if 0 < p < len(s) - 1:
        den = s[p - 1] - 2 * s[p] + s[p + 1]
        if den != 0:
            return 0.5 * (s[p - 1] - s[p + 1]) / den
    return 0.0


# ---------- 歩のタイミング ----------

def detect_steps(tu, sep, fs):
    """足首の離れ具合 sep の山＝1歩。区間(bout)ごとの時刻配列と解析情報を返す。"""
    bouts = []
    for a, b in runs(np.isfinite(sep), int(MIN_BOUT_S * fs)):
        s = lowpass(sep[a:b], fs)
        amp = np.percentile(s, 90) - np.percentile(s, 10)
        if amp <= 0:
            continue
        pk, _ = find_peaks(s, distance=max(1, int(0.25 * fs)), prominence=0.3 * amp)
        if len(pk) < 4:
            continue
        times = np.array([tu[a] + (p + _subframe(s, p)) / fs for p in pk])
        bouts.append({"a": a, "b": b, "s": s, "peaks": pk, "times": times})
    return bouts


def stride_stats(step_times_per_bout, trim=1):
    """歩の時刻から、歩行率・ストライド時間・CV を出す。

    - 各区間の最初と最後の trim 歩は、加速・減速を含むので除く
    - 中央値から大きく外れた歩（検出もれ・二重検出）は除く
    - ストライド＝同じ足の接地から次の接地まで＝2歩ぶん
    """
    steps, strides, rejected = [], [], 0
    for tt in step_times_per_bout:
        tt = np.asarray(tt)
        if len(tt) >= 2 * trim + 3:
            tt = tt[trim:len(tt) - trim]
        if len(tt) < 3:
            continue
        st = np.diff(tt)
        med = np.median(st)
        good = (st > 0.6 * med) & (st < 1.4 * med)
        rejected += int(np.sum(~good))
        steps.extend(st[good])
        for i in range(len(tt) - 2):
            if good[i] and good[i + 1]:
                strides.append(tt[i + 2] - tt[i])
    out = {"n_steps": len(steps), "n_strides": len(strides), "n_rejected": rejected,
           "cadence": np.nan, "mean": np.nan, "sd": np.nan, "cv": np.nan}
    if steps:
        out["cadence"] = 60.0 / float(np.mean(steps))
    if len(strides) >= 2:
        out["mean"] = float(np.mean(strides))
        out["sd"] = float(np.std(strides, ddof=1))
        out["cv"] = out["sd"] / out["mean"] * 100.0
    elif len(strides) == 1:
        out["mean"] = float(strides[0])
    return out


# ---------- 本体 ----------

def _masked_xy(pose: PoseSeries, idx):
    xy = pose.xy[:, idx, :].copy()
    xy[pose.vis[:, idx] < MIN_VIS] = np.nan
    return xy


def analyze(pose: PoseSeries, view: str, distance_m: float = 0.0, time_s: float = 0.0) -> GaitResult:
    """view は 'side'（側面）か 'front'（正面）。distance_m と time_s は手計測（任意）。"""
    res = GaitResult(view=view, fps=pose.fps, detection_rate=pose.detection_rate)
    t = pose.t
    if len(t) < 2 or pose.detection_rate == 0:
        res.messages.append("骨格を検出できませんでした。明るさや、全身が映っているかをご確認ください。")
        return res
    fs = 1.0 / float(np.median(np.diff(t)))

    P = {i: _masked_xy(pose, i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
                                           L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)}
    mid_sh = (P[L_SHOULDER] + P[R_SHOULDER]) / 2
    mid_hip = (P[L_HIP] + P[R_HIP]) / 2
    torso = np.linalg.norm(mid_sh - mid_hip, axis=1)     # 体の大きさの物差し（遠近の補正）

    if view == "side":
        sep_raw = np.abs(P[L_ANKLE][:, 0] - P[R_ANKLE][:, 0]) / torso
    else:
        sep_raw = np.abs(P[L_ANKLE][:, 1] - P[R_ANKLE][:, 1]) / torso
        # 方向転換中（肩幅が細く映る）は除く
        sh_w = np.abs(P[L_SHOULDER][:, 0] - P[R_SHOULDER][:, 0]) / torso
        if np.any(np.isfinite(sh_w)):
            sep_raw[sh_w < 0.6 * np.nanmedian(sh_w)] = np.nan

    tu, sep = resample_uniform(t, sep_raw, fs)
    bouts = detect_steps(tu, sep, fs)
    st = stride_stats([b["times"] for b in bouts])
    res.n_steps, res.n_strides, res.n_rejected_steps = st["n_steps"], st["n_strides"], st["n_rejected"]
    res.cadence_spm, res.stride_time_s = st["cadence"], st["mean"]
    res.stride_time_sd_s, res.stride_time_cv_pct = st["sd"], st["cv"]

    if not bouts:
        res.messages.append("歩行の周期を検出できませんでした。もう少し長く、まっすぐ歩いた動画でお試しください。")
    if view == "side" and bouts:
        _side_metrics(res, pose, P, mid_sh, mid_hip, tu, fs, bouts)

    if distance_m > 0 and time_s > 0:
        res.speed_m_s = distance_m / time_s
        if np.isfinite(res.stride_time_s):
            res.stride_length_m = res.speed_m_s * res.stride_time_s
    return res


def _side_metrics(res, pose, P, mid_sh, mid_hip, tu, fs, bouts):
    t = pose.t

    def rs(y):
        return resample_uniform(t, y, fs)[1]

    # カメラ側の脚＝ visibility の高い側
    vis_l = np.nanmean(pose.vis[:, [L_HIP, L_KNEE, L_ANKLE]])
    vis_r = np.nanmean(pose.vis[:, [R_HIP, R_KNEE, R_ANKLE]])
    near = "left" if vis_l >= vis_r else "right"
    res.near_side = near
    hip_i, knee_i, ank_i, far_ank_i = ((L_HIP, L_KNEE, L_ANKLE, R_ANKLE) if near == "left"
                                       else (R_HIP, R_KNEE, R_ANKLE, L_ANKLE))
    flex = rs(180.0 - interior_angle(P[hip_i], P[knee_i], P[ank_i]))
    near_y, far_y = rs(P[ank_i][:, 1]), rs(P[far_ank_i][:, 1])
    hip_x = rs(mid_hip[:, 0])
    dx = rs(mid_sh[:, 0] - mid_hip[:, 0])
    dy = rs(mid_hip[:, 1] - mid_sh[:, 1])        # 画像は下向きが＋なので、上向きを＋に直す

    leans, knees = [], []
    for b in bouts:
        a, e = b["a"], b["b"]
        hx = hip_x[a:e]
        okx = np.isfinite(hx)
        if okx.sum() < 2:
            continue
        direction = np.sign(hx[okx][-1] - hx[okx][0]) or 1.0     # 進行方向（右へ＝＋）
        lean = np.degrees(np.arctan2(dx[a:e] * direction, dy[a:e]))
        # 最初と最後の歩の間（定常に近い部分）だけ使う
        p0, p1 = b["peaks"][0], b["peaks"][-1]
        leans.extend(lean[p0:p1][np.isfinite(lean[p0:p1])])

        # 立脚中期：左右の足首がすれ違う時点（sep の谷）。カメラ側の足が下にある＝支持脚
        tr, _ = find_peaks(-b["s"], distance=max(1, int(0.25 * fs)))
        for q in tr:
            if not (p0 < q < p1):
                continue
            g = a + q
            if np.isfinite(near_y[g]) and np.isfinite(far_y[g]) and near_y[g] > far_y[g]:
                w = flex[max(g - 1, 0):g + 2]
                if np.any(np.isfinite(w)):
                    knees.append(float(np.nanmedian(w)))
    if leans:
        res.trunk_lean_deg = float(np.mean(leans))
    if knees:
        res.midstance_knee_flexion_deg = float(np.mean(knees))
        res.n_midstance = len(knees)
