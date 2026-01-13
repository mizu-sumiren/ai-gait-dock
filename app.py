import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import base64
from pathlib import Path
import tempfile
import os

# カスタムモジュールのインポート
from female_gait_analyzer import FemaleGaitAnalyzer
from gait_math_core import GaitMathCore
from mediapipe_analyzer import MediaPipeAnalyzer
from video_processor import VideoProcessor

# ページ設定
st.set_page_config(
    page_title="AI歩行ドック - 女性専用分析",
    page_icon="🚺",
    layout="wide"
)

# PWA用設定
def inject_pwa_manifest():
    manifest_path = Path("manifest.json")
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_content = f.read()
        st.markdown(
            f'<link rel="manifest" href="data:application/json;base64,{base64.b64encode(manifest_content.encode()).decode()}">',
            unsafe_allow_html=True
        )

inject_pwa_manifest()

# カスタムCSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    .female-mode-badge {
        background: linear-gradient(135deg, #ff6b9d 0%, #c06c84 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'female_mode' not in st.session_state:
    st.session_state.female_mode = False

st.title("🚺 AI歩行ドック")

# サイドバー
with st.sidebar:
    st.header("📋 患者情報")
    female_mode = st.toggle("🌸 女性専用モード", value=st.session_state.female_mode)
    st.session_state.female_mode = female_mode
    
    age = st.number_input("年齢", 18, 100, 35)
    height = st.number_input("身長 (cm)", 130.0, 200.0, 160.0)
    weight = st.number_input("体重 (kg)", 30.0, 150.0, 55.0)
    
    patient_data = {
        'age': age, 'height': height, 'weight': weight,
        'bmi': weight / ((height / 100) ** 2)
    }

# メインコンテンツ
tab1, tab2 = st.tabs(["📹 動画アップロード", "📊 分析結果"])

with tab1:
    uploaded_file = st.file_uploader("歩行動画をアップロード", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        if st.button("🔍 分析開始", type="primary"):
            with st.spinner("AI分析中..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    video_path = tmp.name
                
                try:
                    analyzer = MediaPipeAnalyzer()
                    pose_data = analyzer.analyze_video(video_path)
                    
                    if pose_data:
                        processor = VideoProcessor()
                        processed_data = processor.process_gait_data(pose_data)
                        
                        math_core = GaitMathCore()
                        results = {'basic_params': math_core.calculate_basic_parameters(processed_data)}
                        
                        if female_mode:
                            f_analyzer = FemaleGaitAnalyzer()
                            results['female_analysis'] = f_analyzer.analyze_female_gait(processed_data, patient_data)
                        
                        st.session_state.analysis_results = results
                        st.session_state.analysis_complete = True
                        st.success("分析完了！")
                finally:
                    if os.path.exists(video_path): os.unlink(video_path)

with tab2:
    if st.session_state.analysis_complete:
        res = st.session_state.analysis_results
        p = res['basic_params']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("歩行速度", f"{p['gait_speed']:.2f} m/s")
        col2.metric("ケイデンス", f"{p['cadence']:.1f} steps/min")
        col3.metric("ステップ長", f"{p['step_length']:.2f} m")
        
        if female_mode and 'female_analysis' in res:
            st.divider()
            st.subheader("🌸 女性専用スコア")
            f = res['female_analysis']
            st.write(f"転倒リスクスコア: {f['fall_risk_score']:.1f}")
            for rec in f['recommendations']:
                st.info(rec)
    else:
        st.info("動画を分析してください。")