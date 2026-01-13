import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from female_gait_analyzer import FemaleGaitAnalyzer

# ページ設定
st.set_page_config(page_title="AI歩行ドック", page_icon="🚺")

st.title("🚺 AI歩行ドック：フェーズ3")
st.write("理学療法士の視点を組み込んだ、あなたのための歩行分析")

# MediaPipeの準備
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ファイルアップローダー
uploaded_file = st.file_uploader("歩行動画（横向き）をアップロードしてください", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # 一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    landmarks_history = []
    
    st.info("解析中... 膝の動きをチェックしています。")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 画像処理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks_history.append(results.pose_landmarks.landmark)
            
            # 描画（簡易版）
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        # 画面にプレビュー表示
        st_frame.image(frame, channels="BGR", use_container_width=True)
        
    cap.release()
    
    # --- 詳細解析の実行 ---
    if landmarks_history:
        analyzer = FemaleGaitAnalyzer()
        clinical_res = analyzer.analyze_clinical_data(landmarks_history)
        
        if clinical_res:
            st.success("解析が完了しました！")
            
            # 結果表示エリア
            st.markdown("---")
            st.header("🏥 理学療法士AIの臨床分析")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("最大膝伸展角度", f"{clinical_res['max_knee_angle']}°")
                # 180度を100%としたプログレスバー
                progress_val = min(clinical_res['max_knee_angle'] / 180.0, 1.0)
                st.progress(progress_val)
                st.caption("180度に近いほど、膝が綺麗に伸びています。")
                
            with col2:
                for msg in clinical_res['recommendations']:
                    if "✨" in msg or "✅" in msg:
                        st.subheader(msg)
                    else:
                        st.write(msg)
    else:
        st.error("骨格が検知できませんでした。もう少し離れて撮影した動画を試してみてください。")

st.sidebar.write("Developed by すみれん | 理学療法士 × AI")
