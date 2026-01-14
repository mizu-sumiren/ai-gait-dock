import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

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

# FemaleGaitAnalyzerのインポート
try:
    from female_gait_analyzer import FemaleGaitAnalyzer
except ImportError:
    st.error("female_gait_analyzer.py が見つかりません。同じディレクトリに配置してください。")
    st.stop()

# ページ設定
st.set_page_config(
    page_title="AI歩行ドック フェーズ3",
    page_icon="🚺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS（ピンク×白基調、清潔感のあるデザイン）
st.markdown("""
<style>
    /* メインヘッダー */
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #E91E63 0%, #F06292 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #757575;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* カード系 */
    .success-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
        border-left: 6px solid #4CAF50;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
        border-left: 6px solid #FFC107;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .danger-card {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 6px solid #F44336;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .clinical-advice-card {
        background: linear-gradient(135deg, #E91E63 0%, #F06292 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(233, 30, 99, 0.3);
    }
    
    .info-card {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-left: 6px solid #2196F3;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* プログレスバー */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #E91E63 0%, #F06292 100%);
    }
    
    /* メトリクス */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #E91E63;
    }
    
    /* サイドバー */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FCE4EC 0%, #F8BBD0 100%);
    }
    
    /* ボタン */
    .stButton > button {
        background: linear-gradient(135deg, #E91E63 0%, #F06292 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(233, 30, 99, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(233, 30, 99, 0.4);
    }
    
    /* ファイルアップローダー */
    div[data-testid="stFileUploader"] {
        background-color: #FFF;
        border: 2px dashed #E91E63;
        border-radius: 12px;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# === 画像前処理関数（CLAHE） ===
def enhance_frame_for_pose_detection(frame):
    """
    CLAHE（適応的ヒストグラム平滑化）によるコントラスト強化
    白背景×白服でも骨格検出を可能にする
    """
    try:
        # RGB変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # LAB色空間に変換してLチャンネルにCLAHE適用
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # 軽いシャープニング
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced_frame, -1, kernel)
        enhanced_frame = cv2.addWeighted(enhanced_frame, 0.7, sharpened, 0.3, 0)
        
        return enhanced_frame
    except:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ヘッダー
st.markdown('<p class="main-header">🚺 AI歩行ドック：フェーズ3</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">理学療法士の視点を組み込んだ、あなたのための歩行分析システム</p>', unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.markdown("### 🩺 AI歩行ドックとは")
    st.write("""
    理学療法士の臨床知識とAI技術を融合させた、
    働く女性のための歩行分析システムです。
    
    **✨ 主な機能:**
    - 歩行周期の自動検出
    - 立脚期の詳細分析
    - 体幹アライメント評価
    - 骨盤底筋リスク評価
    - CLAHE画像強化技術
    """)
    
    st.markdown("---")
    st.markdown("### 🎥 撮影のコツ")
    st.write("""
    **📸 推奨条件:**
    - 明るい自然光の部屋
    - 単色の背景（壁）
    - 服装と背景のコントラスト
    - 完全に横からのアングル
    - 2-5歩程度の自然な歩行
    
    **⚠️ 避けるべき:**
    - 白背景×白い服
    - 暗い環境・逆光
    - 斜めからの撮影
    """)
    
    st.markdown("---")
    st.caption("💖 Developed by すみれん")
    st.caption("理学療法士 × AI Engineer")

# MediaPipe利用可能性チェック
if not MEDIAPIPE_AVAILABLE:
    st.error("⚠️ MediaPipeが正しくインストールされていません。")
    st.stop()

# ファイルアップローダー
st.markdown("### 📹 歩行動画のアップロード")
uploaded_file = st.file_uploader(
    "横から撮影した歩行動画をアップロードしてください",
    type=['mp4', 'mov', 'avi'],
    help="スマートフォンで横向きに撮影した動画が最適です"
)

if uploaded_file is not None:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name
    
    try:
        cap = cv2.VideoCapture(temp_video_path)
        
        if not cap.isOpened():
            st.error("❌ 動画ファイルを開けませんでした。")
            st.stop()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if total_frames == 0:
            st.error("❌ 動画フレーム数が0です。")
            cap.release()
            st.stop()
        
        # MediaPipe Pose初期化（Streamlit Cloud対応設定）
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,  # Streamlit Cloud対応
            smooth_landmarks=True
        ) as pose:
            
            landmarks_history = []
            frame_count = 0
            detection_count = 0
            
            # プレビュー表示
            col1, col2 = st.columns([2, 1])
            with col1:
                st_frame = st.empty()
            with col2:
                st.markdown("#### 📊 処理状況")
                frame_info = st.empty()
                landmark_info = st.empty()
                detection_rate_display = st.empty()
            
            status_text.info("🔍 AI理学療法士が動画を解析中...")
            
            DISPLAY_INTERVAL = 10
            PREVIEW_WIDTH = 640
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                # フレーム存在チェック（二重確認）
                if not ret or frame is None:
                    break
                
                if frame.size == 0:
                    continue
                
                frame_count += 1
                progress = frame_count / total_frames if total_frames > 0 else 0
                progress_bar.progress(min(progress, 1.0))
                
                try:
                    # CLAHE画像強化
                    frame_enhanced = enhance_frame_for_pose_detection(frame)
                    
                    # MediaPipe処理
                    results = pose.process(frame_enhanced)
                    
                    if results.pose_landmarks:
                        landmarks_history.append(results.pose_landmarks.landmark)
                        detection_count += 1
                        
                        # ランドマーク描画
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                        
                        current_detection_rate = (detection_count / frame_count) * 100
                        landmark_info.success(f"✅ 骨格検出: {len(landmarks_history)} フレーム")
                        detection_rate_display.metric(
                            "検出成功率",
                            f"{current_detection_rate:.1f}%",
                            delta=f"{detection_count}/{frame_count}"
                        )
                    else:
                        current_detection_rate = (detection_count / frame_count) * 100
                        landmark_info.warning("⚠️ 骨格未検出")
                        detection_rate_display.metric(
                            "検出成功率",
                            f"{current_detection_rate:.1f}%",
                            delta=f"{detection_count}/{frame_count}"
                        )
                    
                    frame_info.metric("処理フレーム", f"{frame_count}/{total_frames}")
                    
                    # 10フレームごとにリサイズ表示
                    if frame_count % DISPLAY_INTERVAL == 0:
                        height, width = frame.shape[:2]
                        if width > PREVIEW_WIDTH:
                            scale = PREVIEW_WIDTH / width
                            new_width = PREVIEW_WIDTH
                            new_height = int(height * scale)
                            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        else:
                            frame_resized = frame
                        
                        try:
                            st_frame.image(frame_resized, channels="BGR", use_container_width=True)
                        except:
                            pass
                
                except:
                    continue
            
            # 最終フレーム表示
            try:
                if frame is not None and frame.size > 0:
                    height, width = frame.shape[:2]
                    if width > PREVIEW_WIDTH:
                        scale = PREVIEW_WIDTH / width
                        frame_resized = cv2.resize(frame, (PREVIEW_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)
                    else:
                        frame_resized = frame
                    st_frame.image(frame_resized, channels="BGR", use_container_width=True)
            except:
                pass
            
            cap.release()
            
            final_detection_rate = detection_count / total_frames if total_frames > 0 else 0
            status_text.success(f"✅ 処理完了: {len(landmarks_history)}フレーム検出（検出率: {final_detection_rate*100:.1f}%）")
            
            # === 臨床分析の実行 ===
            if len(landmarks_history) >= 30:
                st.markdown("---")
                status_text.info("🧠 AI理学療法士が詳細分析中...")
                
                with st.spinner("歩行パターンを解析しています..."):
                    analyzer = FemaleGaitAnalyzer()
                    clinical_res = analyzer.analyze_clinical_data(landmarks_history)
                
                if clinical_res.get('error'):
                    st.error(f"❌ {clinical_res['error']}: {clinical_res['message']}")
                else:
                    status_text.success("✨ 分析完了！")
                    
                    # === 結果表示 ===
                    st.markdown("---")
                    st.markdown("## 🏥 AI理学療法士の臨床分析レポート")
                    
                    # リスクレベルバッジ
                    risk_level = clinical_res.get('risk_level', 'unknown')
                    risk_badges = {
                        'low': ('🌟 優良', 'success-card'),
                        'moderate': ('💚 良好', 'warning-card'),
                        'high': ('🔔 改善推奨', 'danger-card')
                    }
                    badge_text, card_class = risk_badges.get(risk_level, ('❓ 不明', 'info-card'))
                    st.markdown(f'<div class="{card_class}"><h2 style="margin:0;">{badge_text}</h2></div>', unsafe_allow_html=True)
                    
                    # === メトリクス表示 ===
                    if clinical_res.get('analysis_type') == 'advanced':
                        st.markdown("### 📊 主要な臨床指標")
                        
                        knee_metrics = clinical_res['knee_metrics']
                        trunk_metrics = clinical_res.get('trunk_metrics')
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            stance_ext = knee_metrics['mean_stance_extension']
                            delta_knee = stance_ext - 175.0
                            st.metric(
                                "立脚期 平均膝伸展",
                                f"{stance_ext}°",
                                f"{delta_knee:+.1f}° (理想値との差)",
                                delta_color="normal" if delta_knee >= 0 else "inverse"
                            )
                            progress_val = min(stance_ext / 180.0, 1.0)
                            st.progress(progress_val)
                            st.caption("理想値: 175°以上")
                        
                        with col2:
                            consistency = knee_metrics['consistency']
                            st.metric(
                                "歩行の一貫性",
                                f"±{consistency}°",
                                "安定" if consistency < 5.0 else "ばらつきあり",
                                delta_color="normal" if consistency < 5.0 else "inverse"
                            )
                            st.caption("値が小さいほど安定")
                        
                        with col3:
                            if trunk_metrics and trunk_metrics['mean_trunk_angle'] is not None:
                                trunk_angle = trunk_metrics['mean_trunk_angle']
                                delta_trunk = trunk_angle - 5.0
                                st.metric(
                                    "体幹傾斜角度",
                                    f"{trunk_angle}°",
                                    f"{delta_trunk:+.1f}° (理想値との差)",
                                    delta_color="normal" if delta_trunk <= 0 else "inverse"
                                )
                                st.caption("理想値: 5°以下")
                            else:
                                st.metric("体幹傾斜角度", "測定不可", "データ不足")
                        
                        st.info(f"🚶‍♀️ 検出された歩行周期: **{clinical_res['gait_cycles_detected']}周期**")
                        
                        # === 歩行波形グラフ（Plotly） ===
                        st.markdown("### 📈 歩行パターンの可視化")
                        
                        raw_data = clinical_res.get('raw_data', {})
                        if raw_data:
                            fig = go.Figure()
                            
                            # 生データ
                            fig.add_trace(go.Scatter(
                                y=raw_data['knee_angles_series'],
                                mode='lines',
                                name='生データ',
                                line=dict(color='lightblue', width=1),
                                opacity=0.5
                            ))
                            
                            # 平滑化データ
                            fig.add_trace(go.Scatter(
                                y=raw_data['smoothed_angles'],
                                mode='lines',
                                name='平滑化データ',
                                line=dict(color='#E91E63', width=2)
                            ))
                            
                            # ピーク（立脚期）
                            fig.add_trace(go.Scatter(
                                x=raw_data['peaks'],
                                y=[raw_data['smoothed_angles'][i] for i in raw_data['peaks']],
                                mode='markers',
                                name='立脚期（膝伸展）',
                                marker=dict(color='green', size=10, symbol='star')
                            ))
                            
                            # 谷（遊脚期）
                            fig.add_trace(go.Scatter(
                                x=raw_data['troughs'],
                                y=[raw_data['smoothed_angles'][i] for i in raw_data['troughs']],
                                mode='markers',
                                name='遊脚期（膝屈曲）',
                                marker=dict(color='red', size=10, symbol='circle')
                            ))
                            
                            # 理想値ライン
                            fig.add_hline(y=175.0, line_dash="dash", line_color="green", 
                                        annotation_text="理想値: 175°", annotation_position="right")
                            fig.add_hline(y=165.0, line_dash="dash", line_color="orange",
                                        annotation_text="リスク閾値: 165°", annotation_position="right")
                            
                            fig.update_layout(
                                height=400,
                                xaxis_title="フレーム番号",
                                yaxis_title="膝関節角度（度）",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.caption("""
                            **グラフの見方:**
                            - 🟢 緑の星: 立脚期（膝伸展のピーク）
                            - 🔴 赤の丸: 遊脚期（膝屈曲）
                            - 緑の破線: 理想値（175°）
                            - オレンジの破線: リスク閾値（165°）
                            """)
                        
                        # === データテーブル ===
                        with st.expander("📋 詳細データを確認"):
                            data_dict = {
                                '指標': [
                                    '立脚期平均膝伸展',
                                    '最大膝伸展',
                                    '歩行の一貫性（SD）',
                                    '体幹傾斜角度',
                                    '検出歩行周期数'
                                ],
                                '測定値': [
                                    f"{knee_metrics['mean_stance_extension']}°",
                                    f"{knee_metrics['mean_peak_extension']}°",
                                    f"±{knee_metrics['consistency']}°",
                                    f"{trunk_metrics['mean_trunk_angle']}°" if trunk_metrics else "N/A",
                                    f"{clinical_res['gait_cycles_detected']}周期"
                                ],
                                '評価基準': [
                                    '175°以上が理想',
                                    '180°に近いほど良好',
                                    '5°以下が安定',
                                    '5°以下が理想',
                                    '3周期以上で信頼性向上'
                                ]
                            }
                            df = pd.DataFrame(data_dict)
                            st.dataframe(df, use_container_width=True)
                    
                    else:  # simple分析
                        st.warning("⚠️ 歩行周期が検出できなかったため、簡易分析を表示")
                        max_angle = clinical_res.get('max_knee_angle', 0)
                        st.metric("最大膝伸展角度", f"{max_angle}°")
                        st.progress(min(max_angle / 180.0, 1.0))
                    
                    # === 理学療法士からのアドバイス ===
                    st.markdown("---")
                    st.markdown("## 💬 理学療法士AIからのアドバイス")
                    
                    recommendations = clinical_res.get('recommendations', [])
                    
                    if recommendations:
                        # Markdown表示
                        for rec in recommendations:
                            st.markdown(rec)
                    
                    # === アクションボタン ===
                    st.markdown("---")
                    st.markdown("### 🎯 次のステップ")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📊 Sakane 2025データ", use_container_width=True):
                            sakane_data = analyzer.export_for_sakane_model(clinical_res)
                            if sakane_data:
                                st.json(sakane_data)
                    
                    with col2:
                        if st.button("📥 レポート保存", use_container_width=True):
                            st.info("PDF出力機能は次回実装予定")
                    
                    with col3:
                        if st.button("🔄 別の動画を分析", use_container_width=True):
                            st.rerun()
            
            else:
                st.error(f"❌ 骨格データ不足: {len(landmarks_history)}フレーム（最低30フレーム必要）")
                
                # 臨床的アドバイス
                st.markdown('<div class="clinical-advice-card">', unsafe_allow_html=True)
                st.markdown("### 🩺 理学療法士からのアドバイス")
                st.markdown("""
                **現在の状況:**  
                動画から十分な骨格データを取得できませんでした。
                
                **考えられる原因:**
                1. **背景と服装の色が似ている**（白背景×白服など）
                2. **照明が不十分**（暗い環境・逆光）
                3. **撮影アングルの問題**（斜めから・身体が切れている）
                
                **すぐにできる対策:**
                - ✅ 濃い色のカーディガンを羽織る
                - ✅ 部屋の照明をすべて点ける
                - ✅ 白い壁の前なら黒や紺の服装に
                - ✅ スマホを三脚で真横に固定
                
                正確な歩行分析には、AIがあなたの動きを継続的に追跡できることが重要です。
                検出率70%以上を目指して、再撮影をお試しください。
                """)
                st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ エラーが発生: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        try:
            if 'cap' in locals():
                cap.release()
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        except:
            pass

else:
    # 初期画面
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
        - **背景と服装のコントラスト重要**
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ 動画をアップロード
        - 上のボタンから選択
        - mp4, mov, avi対応
        - iPhone/Android対応
        - CLAHE画像強化で自動処理
        """)
    
    with col3:
        st.markdown("""
        ### 3️⃣ 結果を確認
        - AI理学療法士が分析
        - 詳細なアドバイス
        - Plotlyグラフで可視化
        - Sakane 2025モデル準拠
        """)
    
    st.markdown("---")
    st.info("💡 まずは上のボタンから歩行動画をアップロードしてください")

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #757575; padding: 2rem;'>
    <p style='font-size: 1.2rem; font-weight: bold; color: #E91E63;'>AI歩行ドック フェーズ3</p>
    <p>Powered by MediaPipe × CLAHE × 理学療法士の臨床知識</p>
    <p>Developed by すみれん | Physical Therapist × AI Engineer</p>
    <p style='font-size: 0.8rem; margin-top: 1rem; color: #999;'>
        ⚠️ 本システムは医療診断を目的としたものではありません。<br>
        気になる症状がある場合は、医療機関を受診してください。
    </p>
</div>
""", unsafe_allow_html=True)
