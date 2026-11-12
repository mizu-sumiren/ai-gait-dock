"""画面のスモークテスト（MediaPipe なしで動く部分）。"""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_start_screen_runs():
    at = AppTest.from_file("../app.py").run(timeout=30)
    assert not at.exception
    assert "AI歩行ドック" in at.title[0].value


def _render():
    import app
    from gait.analysis import analyze
    from tests.synthetic import side_walker
    res = analyze(side_walker(fps=30), "side", distance_m=5, time_s=4)
    prev = {**res.to_row(), "trunk_lean_deg": res.trunk_lean_deg - 2}
    app.render_result(res, prev)
    assert app.result_csv(res).startswith(b"\xef\xbb\xbfdate,")


def test_render_result_has_no_risk_wording():
    at = AppTest.from_function(_render).run(timeout=30)
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "体幹の前傾（参考）" in labels and "歩行速度（手計測）" in labels
    lean = next(m for m in at.metric if m.label == "体幹の前傾（参考）")
    assert "+2.0" in lean.delta
    text = " ".join([str(x.value) for x in list(at.info) + list(at.caption) + list(at.markdown)])
    assert "リスク判定" not in text.replace("リスクの判定ではなく", "") and "High Risk" not in text
