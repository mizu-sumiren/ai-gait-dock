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
    st.image("https://via.placeholder.com/150x150.png?text=PT+AI", width=150)
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
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
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                
                # RGB変換
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
                
                # プレビュー表示（10フレームごと）
                if frame_count % 10 == 0:
                    st_frame.image(frame, channels="BGR", use_container_width=True)
            
            cap.release()
            
            # 最終プレビュー表示
            st_frame.image(frame, channels="BGR", use_container_width=True)
            
            status_text.success(f"✅ 動画処理完了: {len(landmarks_history)} フレームの骨格データを取得")
            
            # --- 臨床分析の実行 ---
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
                    
                    # === 結果表示セクション ===
                    st.markdown("---")
                    st.markdown("## 🏥 AI理学療法士の臨床分析レポート")
                    
                    # リスクレベルに応じたバッジ表示
                    risk_level = clinical_res.get('risk_level', 'unknown')
                    risk_badges = {
                        'low': ('🌟 優良', 'success-card'),
                        'moderate': ('💚 良好', 'warning-card'),
                        'high': ('🔔 改善推奨', 'danger-card'),
                        'unknown': ('❓ 不明', 'recommendation-card')
                    }
                    badge_text, card_class = risk_badges.get(risk_level, risk_badges['unknown'])
                    
                    st.markdown(f'<div class="{card_class}"><h3>{badge_text}</h3></div>', unsafe_allow_html=True)
                    
                    # === 主要メトリクスの表示 ===
                    st.markdown("### 📊 主要な臨床指標")
                    
                    if clinical_res.get('analysis_type') == 'advanced':
                        knee_metrics = clinical_res['knee_metrics']
                        trunk_metrics = clinical_res.get('trunk_metrics')
                        
                        # メトリクス表示（3列）
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            stance_extension = knee_metrics['mean_stance_extension']
                            delta_knee = stance_extension - 175.0
                            st.metric(
                                "立脚期 平均膝伸展",
                                f"{stance_extension}°",
                                f"{delta_knee:+.1f}° (理想値との差)",
                                delta_color="normal" if delta_knee >= 0 else "inverse"
                            )
                            
                            # プログレスバー
                            progress_val = min(stance_extension / 180.0, 1.0)
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
                            st.caption("値が小さいほど安定した歩行")
                        
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
                                st.caption("体幹の評価ができませんでした")
                        
                        # 検出された歩行周期数
                        st.info(f"🚶‍♀️ 検出された歩行周期: **{clinical_res['gait_cycles_detected']}周期** （約{clinical_res['gait_cycles_detected']}歩）")
                        
                        # === 歩行波形のグラフ表示 ===
                        st.markdown("### 📈 歩行パターンの可視化")
                        
                        raw_data = clinical_res.get('raw_data', {})
                        if raw_data:
                            # Plotlyで美しいグラフを作成
                            fig = make_subplots(
                                rows=1, cols=1,
                                subplot_titles=("膝関節角度の時系列変化",)
                            )
                            
                            # 生データ（薄い色）
                            fig.add_trace(
                                go.Scatter(
                                    y=raw_data['knee_angles_series'],
                                    mode='lines',
                                    name='生データ',
                                    line=dict(color='lightblue', width=1),
                                    opacity=0.5
                                )
                            )
                            
                            # 平滑化データ（濃い色）
                            fig.add_trace(
                                go.Scatter(
                                    y=raw_data['smoothed_angles'],
                                    mode='lines',
                                    name='平滑化データ',
                                    line=dict(color='blue', width=2)
                                )
                            )
                            
                            # ピーク（立脚期）をマーク
                            fig.add_trace(
                                go.Scatter(
                                    x=raw_data['peaks'],
                                    y=[raw_data['smoothed_angles'][i] for i in raw_data['peaks']],
                                    mode='markers',
                                    name='立脚期（膝伸展）',
                                    marker=dict(color='green', size=10, symbol='star')
                                )
                            )
                            
                            # 谷（遊脚期）をマーク
                            fig.add_trace(
                                go.Scatter(
                                    x=raw_data['troughs'],
                                    y=[raw_data['smoothed_angles'][i] for i in raw_data['troughs']],
                                    mode='markers',
                                    name='遊脚期（膝屈曲）',
                                    marker=dict(color='red', size=10, symbol='circle')
                                )
                            )
                            
                            # 理想値ラインを追加
                            fig.add_hline(
                                y=175.0,
                                line_dash="dash",
                                line_color="green",
                                annotation_text="理想値: 175°",
                                annotation_position="right"
                            )
                            
                            fig.add_hline(
                                y=165.0,
                                line_dash="dash",
                                line_color="orange",
                                annotation_text="リスク閾値: 165°",
                                annotation_position="right"
                            )
                            
                            fig.update_layout(
                                height=400,
                                xaxis_title="フレーム番号",
                                yaxis_title="膝関節角度（度）",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.caption("""
                            **グラフの見方:**
                            - 🟢 緑の星マーク: 立脚期（足が地面についている時の膝の伸び）
                            - 🔴 赤の丸マーク: 遊脚期（足が地面から離れている時の膝の曲がり）
                            - 緑の破線: 理想的な膝伸展角度（175度以上）
                            - オレンジの破線: 改善が推奨される閾値（165度）
                            """)
                        
                        # === データテーブル表示 ===
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
                    
                    else:  # フォールバック表示（simple分析）
                        st.warning("⚠️ 歩行周期が検出できなかったため、簡易分析を表示しています")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            max_angle = clinical_res.get('max_knee_angle', 0)
                            st.metric("最大膝伸展角度", f"{max_angle}°")
                            progress_val = min(max_angle / 180.0, 1.0)
                            st.progress(progress_val)
                            st.caption("180度に近いほど良好")
                        
                        with col2:
                            st.info("より長い距離を自然に歩いている動画で再測定することをお勧めします")
                    
                    # === 理学療法士からのアドバイス ===
                    st.markdown("---")
                    st.markdown("## 💬 理学療法士AIからのアドバイス")
                    
                    recommendations = clinical_res.get('recommendations', [])
                    
                    if recommendations:
                        # recommendationsをMarkdownとして表示
                        full_text = "\n\n".join(recommendations)
                        
                        # カード形式で表示
                        if risk_level == 'low':
                            st.markdown(f'<div class="success-card">{full_text}</div>', unsafe_allow_html=True)
                        elif risk_level == 'moderate':
                            st.markdown(f'<div class="warning-card">{full_text}</div>', unsafe_allow_html=True)
                        elif risk_level == 'high':
                            st.markdown(f'<div class="danger-card">{full_text}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="recommendation-card">{full_text}</div>', unsafe_allow_html=True)
                        
                        # Markdownレンダリング用に再表示（リスト・太字などを反映）
                        with st.container():
                            for rec in recommendations:
                                st.markdown(rec)
                    
                    # === アクションボタン ===
                    st.markdown("---")
                    st.markdown("### 🎯 次のステップ")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📥 レポートをダウンロード", use_container_width=True):
                            # PDFエクスポート機能（今後実装）
                            st.info("PDF出力機能は次回アップデートで実装予定です")
                    
                    with col2:
                        if st.button("📊 Sakane 2025モデル出力", use_container_width=True):
                            sakane_data = analyzer.export_for_sakane_model(clinical_res)
                            if sakane_data:
                                st.json(sakane_data)
                            else:
                                st.warning("Sakane 2025モデル用データが生成できませんでした")
                    
                    with col3:
                        if st.button("🔄 別の動画を分析", use_container_width=True):
                            st.rerun()
                    
            else:
                st.error(f"❌ 骨格データ不足: {len(landmarks_history)}フレーム（最低30フレーム必要）")
                st.info("💡 改善案: より長い距離を歩いている動画（3-5歩以上）を撮影してください")
    
    except Exception as e:
        st.error(f"❌ エラーが発生しました: {str(e)}")
        st.exception(e)
    
    finally:
        # 一時ファイルのクリーンアップ
        import os
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

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
