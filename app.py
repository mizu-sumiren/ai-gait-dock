import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# MediaPipeの堅牢なインポート
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    st.error(f"MediaPipeのインポートに失敗しました: {e}")
    MEDIAPIPE_AVAILABLE = False

from female_gait_analyzer import FemaleGaitAnalyzer

# ページ設定
st.set_page_config(
    page_title="AI歩行ドック フェーズ3",
    page_icon="🚺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS（温かみのあるデザイン）
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E91E63;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #757575;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-card {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-card {
        background-color: #FFF3E0;
        border-left: 5px solid #FFC107;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .danger-card {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ヘッダー
st.markdown('<p class="main-header">🚺 AI歩行ドック：フェーズ3</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">理学療法士の視点を組み込んだ、あなたのための歩行分析</p>', unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.markdown("### 🩺 AI歩行ドックとは")
    st.write("""
    理学療法士の臨床知識とAI技術を融合させた、
    働く女性のための歩行分析システムです。
    
    **特徴:**
    - 歩行周期の自動検出
    - 立脚期の詳細分析
    - 体幹アライメント評価
    - 骨盤底筋リスク評価
    """)
    
    st.markdown("---")
    st.markdown("### 📋 使い方")
    st.write("""
    1. 横から撮影した歩行動画を用意
    2. 動画をアップロード
    3. AIが自動解析
    4. 理学療法士からのアドバイスを確認
    """)
    
    st.markdown("---")
    st.caption("Developed by すみれん | 理学療法士 × AI")

# MediaPipe利用可能性チェック
if not MEDIAPIPE_AVAILABLE:
    st.error("⚠️ MediaPipeが正しくインストールされていません。pip install mediapipe を実行してください。")
    st.stop()

# ファイルアップローダー
st.markdown("### 📹 歩行動画のアップロード")
uploaded_file = st.file_uploader(
    "横から撮影した歩行動画をアップロードしてください（2-5歩程度の自然な歩行）",
    type=['mp4', 'mov', 'avi'],
    help="スマートフォンで横向きに撮影した動画が最適です"
)

if uploaded_file is not None:
    # プログレスバーとステータス表示
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name
    
    try:
        # ビデオキャプチャの初期化
        cap = cv2.VideoCapture(temp_video_path)
        
        # 動画ファイルが正しく開けたか確認
        if not cap.isOpened():
            st.error("❌ 動画ファイルを開けませんでした。ファイル形式を確認してください。")
            st.stop()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if total_frames == 0:
            st.error("❌ 動画フレーム数が0です。有効な動画ファイルをアップロードしてください。")
            cap.release()
            st.stop()
        
        # MediaPipe Poseの初期化
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) as pose:
            
            landmarks_history = []
            frame_count = 0
            
            # プレビュー用のプレースホルダー
            col1, col2 = st.columns([2, 1])
            with col1:
                st_frame = st.empty()
            with col2:
                st.markdown("#### 📊 リアルタイム処理")
                frame_info = st.empty()
                landmark_info = st.empty()
            
            status_text.info("🔍 動画を解析中... 骨格を検出しています")
            
            # フレーム処理設定
            DISPLAY_INTERVAL = 10  # 10フレームごとに表示更新（処理軽減）
            PREVIEW_WIDTH = 640    # プレビュー表示の幅（ピクセル）
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                # === 1. フレーム存在チェック（二重確認） ===
                if not ret or frame is None:
                    # 動画終端または読み込みエラー
                    break
                
                # === 2. フレームの有効性チェック ===
                if frame.size == 0:
                    # 空のフレーム（稀に発生）
                    continue
                
                frame_count += 1
                progress = frame_count / total_frames if total_frames > 0 else 0
                progress_bar.progress(min(progress, 1.0))
                
                try:
                    # === 3. 安全なRGB変換 ===
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # MediaPipe処理
                    results = pose.process(frame_rgb)
                    
                    if results.pose_landmarks:
                        landmarks_history.append(results.pose_landmarks.landmark)
                        
                        # ランドマークを描画
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                        
                        landmark_info.success(f"✅ 骨格検出: {len(landmarks_history)} フレーム")
                    else:
                        landmark_info.warning("⚠️ 骨格未検出")
                    
                    # フレーム情報更新
                    frame_info.metric("処理フレーム", f"{frame_count}/{total_frames}")
                    
                    # === 4. 効率的なプレビュー表示 ===
                    if frame_count % DISPLAY_INTERVAL == 0:
                        # フレームをリサイズして表示（処理軽減）
                        height, width = frame.shape[:2]
                        if width > PREVIEW_WIDTH:
                            scale = PREVIEW_WIDTH / width
                            new_width = PREVIEW_WIDTH
                            new_height = int(height * scale)
                            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        else:
                            frame_resized = frame
                        
                        # 安全な画像表示
                        try:
                            st_frame.image(frame_resized, channels="BGR", use_container_width=True)
                        except Exception as img_error:
                            # 画像表示エラーをスキップ（処理は継続）
                            pass  # 静かにスキップ
                
                except cv2.error as cv_error:
                    # OpenCVエラー（稀に発生）
                    continue  # このフレームをスキップして次へ
                
                except Exception as e:
                    # その他の予期せぬエラー
                    continue  # このフレームをスキップして次へ
            
            # === 5. ループ終了後の安全な最終表示 ===
            try:
                if frame is not None and frame.size > 0:
                    # 最後のフレームを表示
                    height, width = frame.shape[:2]
                    if width > PREVIEW_WIDTH:
                        scale = PREVIEW_WIDTH / width
                        new_width = PREVIEW_WIDTH
                        new_height = int(height * scale)
                        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    else:
                        frame_resized = frame
                    
                    st_frame.image(frame_resized, channels="BGR", use_container_width=True)
            except:
                # 最終表示に失敗しても続行
                pass
            
            cap.release()
            
            status_text.success(f"✅ 動画処理完了: {len(landmarks_history)} フレームの骨格データを取得")
            
            # --- 以降、分析処理は前回のコードと同じ ---
            # （臨床分析の実行セクションをここに挿入）
            
            if len(landmarks_history) >= 30:
                st.markdown("---")
                status_text.info("🧠 AI理学療法士が分析中...")
                
                with st.spinner("詳細な歩行パターンを解析しています..."):
                    analyzer = FemaleGaitAnalyzer()
                    clinical_res = analyzer.analyze_clinical_data(landmarks_history)
                
                if clinical_res.get('error'):
                    st.error(f"❌ {clinical_res['error']}: {clinical_res['message']}")
                else:
                    status_text.success("✨ 分析完了！あなたの歩行レポートをご覧ください")
                    
                    # 以降、前回提供したUIコードをそのまま使用
                    # （メトリクス表示、グラフ、アドバイスなど）
                    
                    # === 結果表示セクション（前回のコードをここに挿入） ===
                    # ... (省略: 前回提供したコードと同じ)
                    
            else:
                st.error(f"❌ 骨格データ不足: {len(landmarks_history)}フレーム（最低30フレーム必要）")
                st.info("💡 改善案: より長い距離を歩いている動画（3-5歩以上）を撮影してください")
    
    except Exception as e:
        st.error(f"❌ エラーが発生しました: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        # 一時ファイルのクリーンアップ
        import os
        try:
            if 'cap' in locals():
                cap.release()
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        except:
            pass

else:
    # 初期画面：使い方ガイド
    st.markdown("---")
    st.markdown("## 📖 AI歩行ドックの使い方")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 1️⃣ 動画を撮影
        - 横から全身が映るように
        - 2-5歩程度歩く
        - 自然な歩き方で
        - 明るい場所で撮影
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ 動画をアップロード
        - 上のボタンから選択
        - mp4, mov, avi対応
        - ファイルサイズ制限なし
        - iPhone/Android両対応
        """)
    
    with col3:
        st.markdown("""
        ### 3️⃣ 結果を確認
        - AI理学療法士が分析
        - 詳細なアドバイス
        - グラフで可視化
        - 改善プランの提案
        """)
    
    st.markdown("---")
    st.info("💡 まずは上のボタンから歩行動画をアップロードしてください")

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #757575; padding: 2rem;'>
    <p><strong>AI歩行ドック フェーズ3</strong></p>
    <p>Powered by MediaPipe × 理学療法士の臨床知識</p>
    <p>Developed by すみれん | Physical Therapist × AI Engineer</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        ⚠️ 本システムは医療診断を目的としたものではありません。<br>
        気になる症状がある場合は、医療機関を受診してください。
    </p>
</div>
""", unsafe_allow_html=True)
