"""AI歩行ドック — 歩き方の記録と、前回からの変化の見える化（画面）"""
import datetime
import math
import os
import tempfile

import pandas as pd
import streamlit as st

from gait.analysis import GaitResult, analyze
from gait.reference import REFERENCES

st.set_page_config(page_title="AI歩行ドック", page_icon="🚶", layout="centered")

MIN_STRIDES = REFERENCES["min_strides"]["value"]

# 表示する指標: (キー, ラベル, 単位, 小数桁, 対象の撮り方)
METRICS = [
    ("speed_m_s", "歩行速度（手計測）", "m/秒", 2, ("side", "front")),
    ("cadence_spm", "歩行率", "歩/分", 0, ("side", "front")),
    ("stride_time_s", "ストライド時間", "秒", 2, ("side", "front")),
    ("stride_time_cv_pct", "ストライド時間のばらつき (CV)", "%", 1, ("side", "front")),
    ("stride_length_m", "ストライド長（速度×時間）", "m", 2, ("side", "front")),
    ("trunk_lean_deg", "体幹の前傾（参考）", "°", 1, ("side",)),
    ("midstance_knee_flexion_deg", "立脚中期の膝の曲がり（参考）", "°", 1, ("side",)),
]


def _ok(v):
    return isinstance(v, (int, float)) and not math.isnan(v)


def render_result(res: GaitResult, prev: dict | None = None):
    """結果の表示。判定や色分けはせず、今回の値と前回との差だけを出す。"""
    for m in res.messages:
        st.warning(m)

    st.caption(f"骨格の検出率 {res.detection_rate * 100:.0f}% ・ 動画 {res.fps:.0f}fps ・ "
               f"解析に使えた歩数 {res.n_steps}歩")
    if res.detection_rate < 0.7:
        st.info("検出率がやや低めです。明るい場所で、服と背景の色が違うと検出しやすくなります。")

    shown = [m for m in METRICS if res.view in m[4] and _ok(getattr(res, m[0]))]
    cols = st.columns(2)
    for i, (key, label, unit, nd, _) in enumerate(shown):
        val = getattr(res, key)
        delta = None
        if prev and _ok(prev.get(key)):
            delta = f"{val - prev[key]:+.{nd}f} {unit}（前回比）"
        cols[i % 2].metric(label, f"{val:.{nd}f} {unit}", delta, delta_color="off")

    walked_strides = res.n_steps // 2
    if _ok(res.stride_time_cv_pct):
        if walked_strides < MIN_STRIDES:
            st.info(f"今回のばらつき(CV)は約{walked_strides}ストライド分からの値です。"
                    f"{REFERENCES['min_strides']['note']}"
                    "少なめのときは、参考程度にご覧ください。")
        if res.fps < 50:
            st.info("30fpsの動画は時間の目盛りが粗いため、ばらつき(CV)が実際より大きめに出ることがあります。"
                    "60fps以上での撮影がおすすめです。")
    if res.n_rejected_steps:
        st.caption(f"タイミングが大きく外れた {res.n_rejected_steps} 歩は、検出ミスの可能性があるため除いています。")
    if res.view == "side" and _ok(res.midstance_knee_flexion_deg):
        side = "左" if res.near_side == "left" else "右"
        st.caption(f"膝はカメラ側（{side}脚）の値です（{res.n_midstance}回の平均）。0°がまっすぐで、反張膝は区別できません。")

    with st.expander("数値の見方と出典"):
        st.markdown(
            "- この結果は**診断やリスクの判定ではなく、歩き方の記録**です。"
            "同じ場所・同じ撮り方で撮った前回の結果と比べてご覧ください。\n"
            f"- {REFERENCES['stride_time_cv']['note']}\n  - 出典: {REFERENCES['stride_time_cv']['source']}\n"
            f"- {REFERENCES['pose_accuracy']['note']}\n  - 出典: {REFERENCES['pose_accuracy']['source']}\n"
            f"- {REFERENCES['workers_gait']['note']}\n  - 出典: {REFERENCES['workers_gait']['source']}\n"
            "- 体幹の前傾と膝の曲がりは、腰痛との関係がはっきり示されている指標ではありません。"
            "姿勢に気づくきっかけとしてお使いください。"
        )


def result_csv(res: GaitResult) -> bytes:
    row = {"date": datetime.date.today().isoformat(), **res.to_row()}
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8-sig")


def load_previous(file) -> dict | None:
    try:
        df = pd.read_csv(file)
        return df.iloc[-1].to_dict() if len(df) else None
    except Exception:
        st.warning("前回のCSVを読み込めませんでした。このアプリで保存したCSVをお選びください。")
        return None


def main():
    st.title("🚶 AI歩行ドック")
    st.write("スマホで撮った歩行動画から、歩き方を数値で記録します。前回の結果と並べて、変化を確認できます。")

    with st.expander("📷 撮り方のコツ", expanded=False):
        st.markdown(
            "- スマホは**横向き**で、三脚などに**固定**します\n"
            "- できれば **60fps** で撮影します（iPhoneは 設定→カメラ→ビデオ撮影）\n"
            "- 全身が映るようにして、明るい場所で撮ります\n"
            "- **正面**：廊下の端にスマホを置き、5〜10mを2〜3往復（リズムのばらつきを見る）\n"
            "- **側面**：真横から3〜4m離して、前を横切るように歩く（体幹・膝の参考値）\n"
            "- 歩行速度は、床に目印を2つ置いて（例：5m）ストップウォッチで測った値を入力します"
        )

    view_label = st.radio("撮った向き", ["正面（前から）", "側面（横から）"], horizontal=True)
    view = "front" if view_label.startswith("正面") else "side"

    c1, c2 = st.columns(2)
    distance = c1.number_input("歩いた距離（m・任意）", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
    seconds = c2.number_input("かかった時間（秒・任意）", min_value=0.0, max_value=120.0, value=0.0, step=0.1)

    video = st.file_uploader("歩行動画", type=["mp4", "mov", "m4v", "avi"])
    prev_file = st.file_uploader("前回の結果CSV（任意・比べたいとき）", type=["csv"])

    if video is not None:
        key = (video.name, video.size, view)
        if st.session_state.get("pose_key") != key:      # 同じ動画は再解析しない
            suffix = os.path.splitext(video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(video.getbuffer())
                path = f.name
            try:
                from gait.pose import extract_pose       # MediaPipe は解析時にだけ読み込む
                bar = st.progress(0.0, text="動画を解析しています…")
                pose = extract_pose(path, progress=lambda p: bar.progress(p, text="動画を解析しています…"))
                bar.empty()
                st.session_state["pose"], st.session_state["pose_key"] = pose, key
            except Exception as e:
                st.error(f"動画の解析でエラーが起きました: {e}")
                return
            finally:
                if os.path.exists(path):
                    os.unlink(path)

        res = analyze(st.session_state["pose"], view, distance_m=distance, time_s=seconds)
        prev = load_previous(prev_file) if prev_file is not None else None
        st.subheader("今回の結果")
        render_result(res, prev)
        st.download_button("📥 結果をCSVで保存（次回の比較用）", result_csv(res),
                           file_name=f"gait_{view}_{datetime.date.today().isoformat()}.csv", mime="text/csv")

    st.divider()
    st.caption("本アプリは医療機器ではなく、診断を目的としたものではありません。"
               "痛みや気になる症状があるときは、医療機関にご相談ください。")


if __name__ == "__main__":
    main()
