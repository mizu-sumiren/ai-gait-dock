import numpy as np
import pytest
from gait.analysis import analyze, interior_angle, lowpass, stride_stats, resample_uniform
from gait.pose import to_pixels
from tests.synthetic import side_walker, front_walker


def test_aspect_ratio_is_corrected():
    # 縦動画(1080x1920)で、本当は170°の膝。正規化座標のままだと約162.6°になっていた
    W, H = 1080, 1920
    hip, knee = np.array([540, 800.0]), np.array([540, 1200.0])
    ank = knee + 400 * np.array([np.sin(np.radians(10)), np.cos(np.radians(10))])
    norm = np.array([hip, knee, ank]) / [W, H]
    assert interior_angle(*norm) == pytest.approx(162.6, abs=0.1)
    assert interior_angle(*to_pixels(norm, W, H)) == pytest.approx(170.0, abs=1e-6)


def test_lowpass_keeps_edges():
    y = lowpass(np.full(200, 170.0), fs=60)
    assert np.allclose(y, 170.0)            # 以前は端が102°などに落ちていた


def test_resample_keeps_long_gaps_as_nan():
    t = np.arange(0, 3, 1 / 30)
    y = np.sin(t)
    y[30:50] = np.nan                        # 約0.67秒の欠損
    _, yu = resample_uniform(t, y, 30)
    assert np.isnan(yu[35:45]).all()


@pytest.mark.parametrize("fps", [30.0, 60.0])
@pytest.mark.parametrize("near,direction", [("right", 1), ("left", -1)])
def test_side_view_recovers_known_values(fps, near, direction):
    pose = side_walker(fps=fps, stride_s=1.1, lean_deg=6.0, knee_deg=8.0, near=near, direction=direction)
    r = analyze(pose, "side")
    assert r.near_side == near
    assert r.stride_time_s == pytest.approx(1.1, abs=0.02)
    assert r.cadence_spm == pytest.approx(60 / 0.55, abs=2)
    assert r.trunk_lean_deg == pytest.approx(6.0, abs=0.5)     # 進行方向が逆でも前傾は＋
    assert r.midstance_knee_flexion_deg == pytest.approx(8.0, abs=1.0)
    assert r.n_midstance >= 1


def test_trunk_lean_is_not_clamped():
    # 以前は体幹角が必ず170°になっていた
    r = analyze(side_walker(lean_deg=3.0), "side")
    assert r.trunk_lean_deg == pytest.approx(3.0, abs=0.5)
    r = analyze(side_walker(lean_deg=-4.0), "side")
    assert r.trunk_lean_deg == pytest.approx(-4.0, abs=0.5)    # 後傾は－


def test_dropped_frames_do_not_shift_time():
    drop = list(range(40, 46)) + list(range(120, 124))         # 短い未検出
    r = analyze(side_walker(drop_frames=drop), "side")
    assert r.stride_time_s == pytest.approx(1.1, abs=0.02)


@pytest.mark.parametrize("cv", [1.5, 4.0])
def test_front_view_stride_cv(cv):
    pose, durs = front_walker(fps=60, cv_pct=cv, n_strides=30)
    r = analyze(pose, "front")
    true_cv = np.std(durs, ddof=1) / np.mean(durs) * 100
    assert r.n_strides >= 15
    assert r.stride_time_s == pytest.approx(np.mean(durs), abs=0.02)
    assert r.stride_time_cv_pct == pytest.approx(true_cv, abs=1.0)


def test_approaching_camera_does_not_inflate_cv():
    # 歩行が一定(CV=0)なら、カメラに近づいてもCVはほぼ0のまま
    pose, _ = front_walker(fps=60, cv_pct=0.0, n_strides=20, approach=True)
    assert analyze(pose, "front").stride_time_cv_pct < 0.5


def test_stride_stats_rejects_missed_step():
    tt = np.arange(0, 11) * 0.55
    tt = np.delete(tt, 5)                    # 1歩検出もれ
    s = stride_stats([tt], trim=0)
    assert s["n_rejected"] == 1
    assert s["mean"] == pytest.approx(1.1, abs=1e-6)


def test_speed_and_stride_length():
    r = analyze(side_walker(), "side", distance_m=5.0, time_s=4.0)
    assert r.speed_m_s == pytest.approx(1.25)
    assert r.stride_length_m == pytest.approx(1.25 * 1.1, abs=0.03)


def test_no_detection_gives_message():
    pose = side_walker()
    pose.xy[:] = np.nan
    pose.vis[:] = 0
    r = analyze(pose, "side")
    assert r.messages and np.isnan(r.stride_time_s)
