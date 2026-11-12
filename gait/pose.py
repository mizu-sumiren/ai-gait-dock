"""動画から骨格（MediaPipe Pose）を取り出す。

ポイント
- 座標は「ピクセル」に直して保存する（正規化座標のままだと縦動画で角度がゆがむ）
- 検出できなかったフレームも捨てずに NaN で残す（時間軸を保つ）
- 時刻は動画のタイムスタンプを使う（iPhoneの可変フレームレート対策）
"""
from dataclasses import dataclass
import numpy as np

# MediaPipe Pose のランドマーク番号
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
N_LANDMARKS = 33


@dataclass
class PoseSeries:
    t: np.ndarray        # (N,) 秒
    xy: np.ndarray       # (N, 33, 2) ピクセル。未検出は NaN
    vis: np.ndarray      # (N, 33) visibility。未検出は 0
    width: int
    height: int
    fps: float

    @property
    def detection_rate(self) -> float:
        if len(self.t) == 0:
            return 0.0
        return float(np.mean(np.isfinite(self.xy[:, L_HIP, 0])))


def to_pixels(norm_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """正規化座標(0-1) → ピクセル座標。x は幅、y は高さを掛ける。"""
    return np.asarray(norm_xy, dtype=float) * np.array([width, height], dtype=float)


def extract_pose(video_path, model_complexity=1, max_side=960, progress=None) -> PoseSeries:
    """動画を1フレームずつ読み、PoseSeries を返す。"""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("動画を開けませんでした")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 <= fps <= 480.0):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ts, xys, viss = [], [], []
    width = height = 0
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                h, w = frame.shape[:2]          # 回転後の実サイズ
                width, height = w, h
                scale = max_side / max(h, w)
                if scale < 1.0:                 # 縦横比を保ったまま縮小（速度のため）
                    frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                                       interpolation=cv2.INTER_AREA)
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                xy = np.full((N_LANDMARKS, 2), np.nan)
                vis = np.zeros(N_LANDMARKS)
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    xy = to_pixels([[p.x, p.y] for p in lm], w, h)
                    vis = np.array([p.visibility for p in lm])
                ts.append(msec / 1000.0 if msec and msec > 0 else i / fps)
                xys.append(xy)
                viss.append(vis)
                i += 1
                if progress and total and i % 10 == 0:
                    progress(min(i / total, 1.0))
    finally:
        cap.release()

    t = np.array(ts, dtype=float)
    # タイムスタンプが単調でない動画は、フレーム番号÷fps に切り替える
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        t = np.arange(len(t)) / fps
    return PoseSeries(t=t, xy=np.array(xys).reshape(-1, N_LANDMARKS, 2),
                      vis=np.array(viss).reshape(-1, N_LANDMARKS),
                      width=width, height=height, fps=fps)
