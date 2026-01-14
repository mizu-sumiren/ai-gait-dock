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
    .clinical-advice-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
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

# === 画像前処理関数（新規追加） ===
def enhance_frame_for_pose_detection(frame):
    """
    低コントラスト環境でも骨格検出しやすいように画像を前処理
    
    Parameters:
    -----------
    frame : numpy.ndarray
        入力フレーム（BGR形式）
    
    Returns:
    --------
    enhanced_frame : numpy.ndarray
        強化されたフレーム（RGB形式）
    """
    try:
        # 1. RGB変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. CLAHE（適応的ヒストグラム平滑化）によるコントラスト強化
        # 各チャンネルに対して適用
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHEの適用（L チャンネルのみ）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # チャンネルを結合
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # 3. 軽いシャープニング（エッジを強調）
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced_frame, -1, kernel)
        
        # オリジナルとブレンド（過度なシャープニングを防ぐ）
        enhanced_frame = cv2.addWeighted(enhanced_frame, 0.7, sharpened, 0.3, 0)
        
        # 4. ガンマ補正（明るさ調整）
        # 暗い場合は明るくする
        mean_brightness = np.mean(enhanced_frame)
        if mean_brightness < 100:
            gamma = 1.2
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                            for i in np.arange(0, 256)]).astype("uint8")
            enhanced_frame = cv2.LUT(enhanced_frame, table)
        
        return enhanced_frame
    
    except Exception as e:
        # 前処理失敗時は元のフレームをRGB変換して返す
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# === 臨床的アドバイス生成関数（新規追加） ===
def generate_clinical_shooting_advice(landmarks_history, total_frames, detection_rate):
    """
    骨格検出率に基づいて、理学療法士らしい撮影アドバイスを生成
    
    Parameters:
    -----------
    landmarks_history : list
        検出された骨格データのリスト
    total_frames : int
        総フレーム数
    detection_rate : float
        検出成功率（0.0-1.0）
    
    Returns:
    --------
    advice : dict
        アドバイス情報（メッセージ、重要度など）
    """
    advice = {
        'severity': 'info',  # 'info', 'warning', 'error'
        'title': '',
        'messages': [],
        'tips': []
    }
    
    if detection_rate < 0.1:  # 10%未満
        advice['severity'] = 'error'
        advice['title'] = '🔴 骨格検出率が極めて低い状態です'
        advice['messages'] = [
            f"動画全体で{len(landmarks_history)}フレーム（{detection_rate*100:.1f}%）しか骨格を検出できませんでした。",
            "",
            "**理学療法士からの診断:**",
            "現在の映像では、AIがあなたの身体の輪郭を正確に認識できていません。これは臨床分析に必要な精度を確保できない状態です。",
        ]
        advice['tips'] = [
            "### 📸 撮影環境の改善が必須です",
            "",
            "**最優先の対策:**",
            "1. **背景と服装のコントラスト確保**",
            "   - 白い背景 → 濃い色の服（黒、紺、グレーなど）",
            "   - 暗い背景 → 明るい色の服（白、ベージュなど）",
            "   - 理想: 背景が単色の壁（グレー、ベージュ）で、対照的な服装",
            "",
            "2. **照明の最適化**",
            "   - 自然光が入る明るい部屋で撮影",
            "   - 逆光を避ける（窓を背にしない）",
            "   - 影が強く出ないよう、柔らかい光が理想",
            "",
            "3. **カメラアングルの確認**",
            "   - 完全に真横から撮影（斜めNG）",
            "   - 全身が画面に収まるように",
            "   - 床から天井まで余裕を持たせる",
            "",
            "**臨床検査としての重要性:**",
            "正確な歩行分析には、骨格検出率70%以上が必要です。現在の環境では医療データとして信頼できる結果が得られません。",
        ]
    
    elif detection_rate < 0.3:  # 30%未満
        advice['severity'] = 'warning'
        advice['title'] = '⚠️ 骨格検出率が不十分です'
        advice['messages'] = [
            f"動画全体で{len(landmarks_history)}フレーム（{detection_rate*100:.1f}%）の骨格を検出できました。",
            "",
            "**理学療法士からの所見:**",
            "部分的には検出できていますが、臨床的に信頼できる歩行分析を行うには、より安定した検出が必要です。",
        ]
        advice['tips'] = [
            "### 🎯 検出精度を高めるための対策",
            "",
            "**推奨される改善策:**",
            "1. **服装の見直し**",
            "   - 現在の背景と対照的な色の服に着替える",
            "   - フィット感のある服（ダボダボすぎない）",
            "   - できれば濃い色の長袖・長ズボン",
            "",
            "2. **背景の変更**",
            "   - 無地の壁の前で撮影",
            "   - 家具や装飾品が映り込まないように",
            "   - 床の模様も検出を妨げる可能性あり",
            "",
            "3. **照明条件**",
            "   - より明るい場所を選ぶ",
            "   - 部屋の電気をすべて点ける",
            "   - 日中の自然光を活用",
            "",
            "**目標:** 検出率70%以上で、安定した臨床データが取得できます。",
        ]
    
    elif detection_rate < 0.7:  # 70%未満
        advice['severity'] = 'warning'
        advice['title'] = '💡 検出精度向上の余地があります'
        advice['messages'] = [
            f"動画全体で{len(landmarks_history)}フレーム（{detection_rate*100:.1f}%）の骨格を検出できました。",
            "",
            "**理学療法士からの評価:**",
            "基本的な分析は可能ですが、より高精度なデータを得るために、いくつかの改善ができます。",
        ]
        advice['tips'] = [
            "### ✨ さらに精度を高めるには",
            "",
            "**より良い結果を得るための提案:**",
            "- 背景と服装のコントラストをさらに強調",
            "- 照明をもう少し明るく",
            "- カメラを完全に真横に固定",
            "",
            "現在でも分析可能ですが、上記を試すことで、より詳細な臨床データが得られます。",
        ]
    
    else:  # 70%以上
        advice['severity'] = 'info'
        advice['title'] = '✅ 優れた検出精度です'
        advice['messages'] = [
            f"動画全体で{len(landmarks_history)}フレーム（{detection_rate*100:.1f}%）の骨格を検出できました。",
            "",
            "**理学療法士からの評価:**",
            "臨床分析に十分な精度でデータが取得できています。この品質であれば、信頼性の高い歩行評価が可能です。",
        ]
        advice['tips'] = []
    
    return advice

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
    - 高精度骨格検出（model_complexity=2）
    - 低コントラスト環境対応
    - 歩行周期の自動検出
    - 立脚期の詳細分析
    - 体幹アライメント評価
    - 骨盤底筋リスク評価
    """)
    
    st.markdown("---")
    st.markdown("### 🎥 最適な撮影条件")
    st.write("""
    **推奨環境:**
    - 明るい自然光の部屋
    - 単色の背景（壁）
    - 服装と背景のコントラスト
    - 完全に横からのアングル
    
    **避けるべき条件:**
    - 逆光（窓を背にする）
    - 白背景×白い服
    - 暗い部屋
    - 斜めからの撮影
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
    help="スマートフォンで横向きに撮影した動画が最適です。背景と服装のコントラストを意識してください。"
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
        
        # === 1. MediaPipe Poseの最高性能設定 ===
        with mp_pose.Pose(
            min_detection_confidence=0.4,  # 検出しきい値を下げて感度向上
            min_tracking_confidence=0.4,   # 追跡しきい値を下げて継続性向上
            model_complexity=2,             # 最高精度モデル（重いが正確）
            smooth_landmarks=True,          # ランドマークの平滑化を有効化
            enable_segmentation=False       # セグメンテーションは無効（高速化）
        ) as pose:
            
            landmarks_history = []
            frame_count = 0
            detection_count = 0  # 検出成功カウント
            
            # プレビュー用のプレースホルダー
            col1, col2 = st.columns([2, 1])
            with col1:
                st_frame = st.empty()
            with col2:
                st.markdown("#### 📊 リアルタイム処理")
                frame_info = st.empty()
                landmark_info = st.empty()
                detection_rate_display = st.empty()
            
            status_text.info("🔍 高精度モードで解析中... 骨格を検出しています（処理時間：通常の1.5-2倍）")
            
            # フレーム処理設定
            DISPLAY_INTERVAL = 10  # 10フレームごとに表示更新
            PREVIEW_WIDTH = 640    # プレビュー表示の幅
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                # フレーム存在チェック
                if not ret or frame is None:
                    break
                
                if frame.size == 0:
                    continue
                
                frame_count += 1
                progress = frame_count / total_frames if total_frames > 0 else 0
                progress_bar.progress(min(progress, 1.0))
                
                try:
                    # === 3. 画像前処理（コントラスト強化） ===
                    frame_enhanced = enhance_frame_for_pose_detection(frame)
                    
                    # MediaPipe処理（強化されたフレームを使用）
                    results = pose.process(frame_enhanced)
                    
                    if results.pose_landmarks:
                        landmarks_history.append(results.pose_landmarks.landmark)
                        detection_count += 1
                        
                        # ランドマークを描画（元のフレームに）
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                        
                        # 検出率計算
                        current_detection_rate = (detection_count / frame_count) * 100
                        
                        landmark_info.success(f"✅ 骨格検出: {len(landmarks_history)} フレーム")
                        detection_rate_display.metric(
                            "検出成功率",
                            f"{current_detection_rate:.1f}%",
                            delta=f"{detection_count}/{frame_count}"
                        )
                    else:
                        current_detection_rate = (detection_count / frame_count) * 100
                        landmark_info.warning(f"⚠️ 骨格未検出（フレーム{frame_count}）")
                        detection_rate_display.metric(
                            "検出成功率",
                            f"{current_detection_rate:.1f}%",
                            delta=f"{detection_count}/{frame_count}"
                        )
                    
                    # フレーム情報更新
                    frame_info.metric("処理フレーム", f"{frame_count}/{total_frames}")
                    
                    # 効率的なプレビュー表示
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
                
                except cv2.error:
                    continue
                except Exception:
                    continue
            
            # ループ終了後の最終表示
            try:
                if frame is not None and frame.size > 0:
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
                pass
            
            cap.release()
            
            # === 最終検出率の計算 ===
            final_detection_rate = detection_count / total_frames if total_frames > 0 else 0
            
            status_text.success(f"✅ 動画処理完了: {len(landmarks_history)}フレーム検出（検出率: {final_detection_rate*100:.1f}%）")
            
            # === 4. 臨床的アドバイスの表示 ===
            st.markdown("---")
            st.markdown("## 🎯 骨格検出品質レポート")
            
            # 検出率に基づくアドバイス生成
            advice = generate_clinical_shooting_advice(
                landmarks_history,
                total_frames,
                final_detection_rate
            )
            
            # アドバイスカードの表示
            if advice['severity'] == 'error':
                st.markdown(f'<div class="danger-card"><h3>{advice["title"]}</h3></div>', unsafe_allow_html=True)
            elif advice['severity'] == 'warning':
                st.markdown(f'<div class="warning-card"><h3>{advice["title"]}</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-card"><h3>{advice["title"]}</h3></div>', unsafe_allow_html=True)
            
            # メッセージ表示
            for msg in advice['messages']:
                st.markdown(msg)
            
            # Tips表示
            if advice['tips']:
                with st.expander("📚 詳しい改善アドバイスを見る", expanded=(advice['severity'] == 'error')):
                    for tip in advice['tips']:
                        st.markdown(tip)
            
            # === 分析実行判定 ===
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
                    
                    # ここに前回提供した完全な結果表示UIを挿入
                    # （メトリクス、グラフ、アドバイスなど）
                    st.markdown("---")
                    st.markdown("## 🏥 AI理学療法士の臨床分析レポート")
                    st.info("（ここに前回の結果表示コードが入ります）")
                    
            else:
                st.markdown("---")
                st.error(f"❌ 骨格データ不足: {len(landmarks_history)}フレーム（最低30フレーム必要）")
                
                # 臨床的なアドバイス
                st.markdown('<div class="clinical-advice-card">', unsafe_allow_html=True)
                st.markdown("### 🩺 理学療法士からのアドバイス")
                st.markdown("""
                **現在の状況:**
                動画から十分な骨格データを取得できませんでした。これは以下の理由が考えられます：
                
                1. **背景と服装の色が似ている**
                   - 白い背景に白や明るい色の服装
                   - AIが身体の輪郭を認識できない状態
                
                2. **照明が不十分**
                   - 暗い環境や逆光
                   - 影が強く出ている
                
                3. **撮影アングルの問題**
                   - 完全に横からでない
                   - 身体の一部が切れている
                
                **臨床検査として再撮影をお勧めします:**
                
                ✅ **すぐにできる対策**
                - 濃い色のカーディガンやジャケットを羽織る
                - 白い壁の前なら、黒や紺の服装に
                - 部屋の照明をすべて点ける
                - スマホを三脚や台に固定して真横から
                
                正確な歩行分析には、AIがあなたの動きを継続的に追跡できることが重要です。
                上記の対策で、検出率70%以上を目指しましょう。
                """)
                st.markdown('</div>', unsafe_allow_html=True)
    
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
        - **背景と服装のコントラスト重要**
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ 動画をアップロード
        - 上のボタンから選択
        - mp4, mov, avi対応
        - ファイルサイズ制限なし
        - iPhone/Android両対応
        - 高精度モードで解析
        """)
    
    with col3:
        st.markdown("""
        ### 3️⃣ 結果を確認
        - AI理学療法士が分析
        - 詳細なアドバイス
        - グラフで可視化
        - 改善プランの提案
        - 撮影品質フィードバック
        """)
    
    st.markdown("---")
    
    # 撮影のコツを強調
    st.markdown("### 💡 高精度分析のための撮影のコツ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### ✅ 推奨される条件
        - 単色の壁を背景にする
        - 服装と背景の色を対照的に
        - 明るい自然光の部屋
        - 完全に真横からのアングル
        - 全身が画面に収まるように
        """)
    
    with col2:
        st.markdown("""
        #### ❌ 避けるべき条件
        - 白背景 × 白い服装
        - 複雑な背景（家具・装飾）
        - 暗い部屋・逆光
        - 斜めからの撮影
        - 身体が切れている
        """)
    
    st.info("💡 まずは上のボタンから歩行動画をアップロードしてください")

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #757575; padding: 2rem;'>
    <p><strong>AI歩行ドック フェーズ3 - 高精度骨格検出版</strong></p>
    <p>Powered by MediaPipe（model_complexity=2） × 理学療法士の臨床知識</p>
    <p>CLAHE画像強化 + 低コントラスト環境対応</p>
    <p>Developed by すみれん | Physical Therapist × AI Engineer</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        ⚠️ 本システムは医療診断を目的としたものではありません。<br>
        気になる症状がある場合は、医療機関を受診してください。
    </p>
</div>
""", unsafe_allow_html=True)
