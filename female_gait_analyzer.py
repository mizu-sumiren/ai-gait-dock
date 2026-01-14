import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

class FemaleGaitAnalyzer:
    def __init__(self):
        # 臨床的な閾値（Sakane 2025モデル準拠）- 微調整版
        self.thresholds = {
            'knee_extension_ideal': 175.0,      # 理想値（上方修正）
            'knee_extension_good': 170.0,       # 良好の基準（新設）
            'knee_extension_minimum': 165.0,    # 最低基準
            'stance_phase_mean_minimum': 168.0, # 立脚期平均（170→168に緩和）
            'trunk_alignment_ideal': 5.0,       # 体幹傾斜の理想値（度）
            'trunk_risk_threshold': 15.0,       # 体幹傾斜リスク閾値
        }
        
        # 歩行周期検出のパラメータ
        self.gait_cycle_params = {
            'min_peak_distance': 15,
            'prominence': 5.0,
        }
        
        # 信頼度フィルタのパラメータ（新設）
        self.confidence_params = {
            'min_visibility': 0.7,              # 最低視認性閾値
            'max_angle_change': 8.0,            # 1フレーム間の最大角度変化（度）
            'noise_window': 5,                  # ノイズ平滑化ウィンドウ
        }

    def _calculate_angle(self, a, b, c):
        """3点の座標から角度を計算（180度を最大とする）"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _filter_by_confidence(self, angles, visibilities):
        """
        信頼度フィルタ：低信頼度フレームを補完・除外
        
        Parameters:
        -----------
        angles : list
            角度のリスト
        visibilities : list
            各フレームの視認性スコアのリスト
        
        Returns:
        --------
        filtered_angles : np.ndarray
            フィルタリング後の角度配列
        """
        if len(angles) != len(visibilities):
            return np.array(angles)
        
        angles = np.array(angles)
        visibilities = np.array(visibilities)
        
        # 1. 低信頼度フレームをマスク
        valid_mask = visibilities >= self.confidence_params['min_visibility']
        
        if np.sum(valid_mask) < 10:  # 有効フレームが少なすぎる場合
            return angles  # フィルタリングせずに返す
        
        # 2. 線形補間で低信頼度フレームを埋める
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < len(angles):
            interp_func = interp1d(
                valid_indices, 
                angles[valid_mask], 
                kind='linear', 
                fill_value='extrapolate'
            )
            all_indices = np.arange(len(angles))
            angles = interp_func(all_indices)
        
        # 3. 急激な変化を検出して平滑化
        angle_diffs = np.abs(np.diff(angles))
        noise_mask = angle_diffs > self.confidence_params['max_angle_change']
        
        if np.any(noise_mask):
            # ノイズが検出された場合、移動平均で平滑化
            window = self.confidence_params['noise_window']
            angles = np.convolve(angles, np.ones(window)/window, mode='same')
        
        return angles

    def _detect_gait_cycles(self, knee_angles):
        """
        膝角度の時系列データから歩行周期を検出
        立脚期（膝伸展）と遊脚期（膝屈曲）を判別
        """
        if len(knee_angles) < 30:
            return None
        
        # ノイズ除去（Savitzky-Golayフィルタ）
        window_length = min(15, len(knee_angles) if len(knee_angles) % 2 == 1 else len(knee_angles) - 1)
        if window_length < 5:
            smoothed = knee_angles
        else:
            smoothed = signal.savgol_filter(knee_angles, window_length, 3)
        
        # ピーク検出（膝伸展の最大値 = 立脚中期）
        peaks, properties = signal.find_peaks(
            smoothed,
            distance=self.gait_cycle_params['min_peak_distance'],
            prominence=self.gait_cycle_params['prominence']
        )
        
        # 谷検出（膝屈曲の最大値 = 遊脚期）
        troughs, _ = signal.find_peaks(
            -smoothed,
            distance=self.gait_cycle_params['min_peak_distance'],
            prominence=self.gait_cycle_params['prominence']
        )
        
        if len(peaks) == 0:
            return None
        
        # 歩行周期の定義
        cycles = []
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            
            # この周期内の立脚期を推定（ピークの前後20%）
            cycle_length = end_idx - start_idx
            stance_start = max(0, start_idx - int(cycle_length * 0.2))
            stance_end = min(len(smoothed), start_idx + int(cycle_length * 0.2))
            
            cycles.append({
                'start': start_idx,
                'end': end_idx,
                'peak': start_idx,
                'stance_phase': (stance_start, stance_end),
                'peak_angle': smoothed[start_idx]
            })
        
        return {
            'cycles': cycles,
            'smoothed_angles': smoothed,
            'raw_angles': knee_angles,
            'peaks': peaks,
            'troughs': troughs
        }

    def _calculate_stance_phase_metrics(self, gait_data):
        """立脚期の平均的な膝伸展角度を評価"""
        if not gait_data or not gait_data['cycles']:
            return None
        
        stance_means = []
        stance_maxs = []
        
        for cycle in gait_data['cycles']:
            stance_start, stance_end = cycle['stance_phase']
            stance_angles = gait_data['smoothed_angles'][stance_start:stance_end]
            
            if len(stance_angles) > 0:
                stance_means.append(np.mean(stance_angles))
                stance_maxs.append(np.max(stance_angles))
        
        return {
            'mean_stance_extension': np.mean(stance_means) if stance_means else 0,
            'mean_peak_extension': np.mean(stance_maxs) if stance_maxs else 0,
            'consistency': np.std(stance_means) if len(stance_means) > 1 else 0,
            'num_cycles': len(gait_data['cycles'])
        }

    def _calculate_trunk_alignment(self, landmarks_history):
        """
        体幹の垂直性を評価（修正版）
        垂直線（0度）からの傾斜角度を計算
        
        理学療法士の視点：
        - 垂直 = 0度
        - 前傾 = 正の角度
        - 後傾 = 負の角度（絶対値で評価）
        """
        trunk_angles = []
        visibilities = []
        
        for lm in landmarks_history:
            try:
                # 右側の肩(12)と股関節(24)を使用
                shoulder = np.array([lm[12].x, lm[12].y])
                hip = np.array([lm[24].x, lm[24].y])
                
                # 視認性チェック
                shoulder_vis = lm[12].visibility
                hip_vis = lm[24].visibility
                avg_vis = (shoulder_vis + hip_vis) / 2
                
                # 低信頼度フレームはスキップ
                if avg_vis < 0.5:
                    continue
                
                # 体幹ベクトル（肩→股関節）
                trunk_vector = hip - shoulder
                
                # 垂直ベクトル（下向き = [0, 1]）
                # MediaPipeのy座標は上が0、下が1なので、垂直下向きは[0, 1]
                vertical_vector = np.array([0, 1])
                
                # 内積で角度を計算
                dot_product = np.dot(trunk_vector, vertical_vector)
                trunk_norm = np.linalg.norm(trunk_vector)
                vertical_norm = np.linalg.norm(vertical_vector)
                
                cos_angle = dot_product / (trunk_norm * vertical_norm)
                angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                
                # ラジアンから度に変換
                trunk_angle = np.degrees(angle_rad)
                
                # 前傾・後傾の判定（x座標の差で判断）
                # 肩が股関節より前にある = 前傾 = 正の角度
                # 肩が股関節より後ろ = 後傾 = 負の角度
                if trunk_vector[0] > 0:  # 前傾
                    pass  # そのまま
                else:  # 後傾
                    trunk_angle = -trunk_angle
                
                # 垂直からの絶対偏差を記録（評価用）
                trunk_angles.append(np.abs(trunk_angle))
                visibilities.append(avg_vis)
                
            except Exception as e:
                continue
        
        if not trunk_angles:
            return None
        
        # 信頼度フィルタを適用
        filtered_trunk_angles = self._filter_by_confidence(trunk_angles, visibilities)
        
        return {
            'mean_trunk_angle': np.mean(filtered_trunk_angles),
            'max_trunk_angle': np.max(filtered_trunk_angles),
            'trunk_variability': np.std(filtered_trunk_angles),
            'trunk_angles_series': filtered_trunk_angles.tolist()
        }

    def _generate_clinical_recommendations(self, knee_metrics, trunk_metrics, gait_data):
        """
        理学療法士視点の温かく具体的なフィードバック生成
        女性特有の骨盤底筋・膝OA予防の視点を強化
        """
        recs = []
        risk_level = "low"
        
        # === 膝の評価（閾値微調整版） ===
        mean_stance = knee_metrics['mean_stance_extension']
        mean_peak = knee_metrics['mean_peak_extension']
        consistency = knee_metrics['consistency']
        
        recs.append("### 🚶‍♀️ あなたの歩行パターン分析")
        
        if mean_stance < self.thresholds['stance_phase_mean_minimum']:
            risk_level = "high"
            recs.append(f"**膝の伸びに関する大切なお知らせ**  ")
            recs.append(f"立脚期の平均膝伸展は{round(mean_stance, 1)}度です。")
            recs.append("")
            recs.append("💭 **理学療法士からの気づき**  ")
            recs.append("膝が完全に伸びきらない歩き方は、実は多くの働く女性に見られるパターンです。デスクワークで股関節屈筋群が硬くなったり、ヒールを履く習慣で代償動作が起きたりすることで、無意識に膝を軽く曲げて歩くクセがついていることがあります。")
            recs.append("")
            recs.append("🌸 **女性の身体を守る視点**  ")
            recs.append("- **骨盤底筋への影響**: 膝が伸びないと体幹が前傾し、骨盤底筋に余計な負担がかかります")
            recs.append("- **将来の膝痛リスク**: 40代以降、女性は男性の2倍の確率で変形性膝関節症を発症します")
            recs.append("- **疲労の蓄積**: 夕方になると足が重く感じるのは、この歩き方が原因かもしれません")
            recs.append("")
            recs.append("💡 **今日から始める3つのケア**  ")
            recs.append("1. **デスクでできる股関節ストレッチ**: 椅子に浅く座り、片足を前に伸ばして膝裏を床に近づける（15秒×3回）")
            recs.append("2. **歩行の意識改革**: 「かかと→小指球→親指球」の順で地面を押す感覚を意識")
            recs.append("3. **骨盤底筋トレーニング**: 立脚期に骨盤底を軽く引き上げる意識を持つ")
            
        elif mean_stance < self.thresholds['knee_extension_good']:
            risk_level = "moderate"
            recs.append(f"**膝の伸びは概ね良好です**  ")
            recs.append(f"立脚期の平均膝伸展は{round(mean_stance, 1)}度（良好基準: {self.thresholds['knee_extension_good']}度以上）")
            recs.append("")
            recs.append("✨ **あと一歩で理想的な歩行へ**  ")
            recs.append("現在の歩き方は良い状態です。あと少し膝の伸びを改善することで、以下のメリットが得られます：")
            recs.append("- 長時間歩いても疲れにくい身体")
            recs.append("- 夕方の下半身のむくみ軽減")
            recs.append("- 骨盤底筋の機能維持")
            recs.append("")
            recs.append("🎯 **ワンランク上の歩行へ**  ")
            recs.append("通勤時に「胸を開いて、遠くを見て歩く」ことを意識してみてください。視線が上がると自然と体幹が安定し、膝も伸びやすくなります。")
            
        elif mean_stance < self.thresholds['knee_extension_ideal']:
            risk_level = "low"
            recs.append(f"**膝の伸びは良好です！**  ")
            recs.append(f"立脚期の平均膝伸展は{round(mean_stance, 1)}度。とても良い状態です。")
            recs.append("")
            recs.append("🌟 **理想値まであと少し**  ")
            recs.append(f"理想値の{self.thresholds['knee_extension_ideal']}度まであと{round(self.thresholds['knee_extension_ideal'] - mean_stance, 1)}度です。")
            recs.append("現在の状態を維持しながら、股関節の柔軟性を高めることで、さらに向上します。")
            
        else:
            recs.append(f"**✨ 理想的な膝の伸びです！**  ")
            recs.append(f"立脚期の平均膝伸展は{round(mean_stance, 1)}度。素晴らしい歩行パターンです。")
            recs.append("")
            recs.append("🌟 **この歩き方がもたらす恩恵**  ")
            recs.append("- **骨盤底筋の機能維持**: 体幹が安定し、骨盤底筋への負担が最小限")
            recs.append("- **膝関節の健康**: 将来の変形性膝関節症リスクが低い歩行パターン")
            recs.append("- **生産性向上**: 疲れにくい身体で、午後の集中力も維持")
            recs.append("")
            recs.append("💪 **この状態を維持するために**  ")
            recs.append("定期的な歩行チェックと、股関節周りの柔軟性を保つストレッチを継続してください。")
        
        # === 一貫性の評価 ===
        recs.append("")
        if consistency > 5.0:
            recs.append("⚠️ **歩行の一貫性について**  ")
            recs.append(f"歩行周期ごとの膝の伸びにばらつきが見られます（標準偏差: {round(consistency, 1)}度）。")
            recs.append("これは疲労や痛みの回避、あるいは左右差が原因の可能性があります。")
            recs.append("片側に負担が集中していないか、鏡の前でチェックしてみましょう。")
        else:
            recs.append("✅ **歩行の一貫性**: 素晴らしいです。各歩行周期で安定した動きができています。")
        
        # === 体幹の評価（修正版） ===
        if trunk_metrics:
            recs.append("")
            recs.append("### 🧘‍♀️ 体幹の評価（Sakane 2025モデル）")
            mean_trunk = trunk_metrics['mean_trunk_angle']
            
            if mean_trunk > self.thresholds['trunk_risk_threshold']:
                risk_level = "high" if risk_level != "high" else risk_level
                recs.append(f"**体幹の傾きが気になります**  ")
                recs.append(f"平均体幹傾斜: {round(mean_trunk, 1)}度（理想値: {self.thresholds['trunk_alignment_ideal']}度以内）")
                recs.append("")
                recs.append("🎯 **体幹傾斜と女性の健康**  ")
                recs.append("体幹が前傾すると、骨盤底筋に持続的な下向きの圧力がかかり、将来的な尿もれや骨盤臓器脱のリスクが高まります。")
                recs.append("また、腰痛や肩こりの原因にもなります。")
                recs.append("")
                recs.append("💡 **体幹を整えるアプローチ**  ")
                recs.append("1. **呼吸と体幹の連動**: 鼻から吸って、息を吐くときに骨盤底を引き上げる")
                recs.append("2. **プランク変法**: 膝をついた状態で10秒キープ、デスクワークの合間に")
                recs.append("3. **歩行時の意識**: 「頭が天井から糸で引っ張られている」イメージで")
                
            elif mean_trunk > self.thresholds['trunk_alignment_ideal']:
                recs.append(f"**体幹はほぼ良好ですが、さらなる改善が可能です**  ")
                recs.append(f"平均体幹傾斜: {round(mean_trunk, 1)}度")
                recs.append("デスクワークの姿勢が影響している可能性があります。1時間に1回は立ち上がって背伸びを。")
                
            else:
                recs.append(f"**✨ 理想的な体幹アライメントです！**  ")
                recs.append(f"平均体幹傾斜: {round(mean_trunk, 1)}度")
                recs.append("骨盤底筋への負担が最小限で、長期的な健康維持に理想的な姿勢です。")
        
        # === 総合リスク評価 ===
        recs.append("")
        recs.append("---")
        if risk_level == "high":
            recs.append("### 🔔 総合評価: 改善推奨")
            recs.append("現在の歩行パターンには改善の余地があります。でも大丈夫、小さな意識の変化で確実に変わります。")
            recs.append("**次のステップ**: 上記の3つのケアを2週間続けて、再度測定してみましょう。")
        elif risk_level == "moderate":
            recs.append("### 💚 総合評価: 良好")
            recs.append("現在の状態は良好です。継続的なケアで理想的な歩行パターンを手に入れましょう。")
        else:
            recs.append("### 🌟 総合評価: 優良")
            recs.append("素晴らしい歩行パターンです。この状態を維持することで、生涯にわたる身体の健康が期待できます。")
        
        return recs, risk_level

    def analyze_clinical_data(self, landmarks_history):
        """
        臨床データの総合分析（信頼度フィルタ統合版）
        """
        if not landmarks_history or len(landmarks_history) < 30:
            return {
                'error': 'データ不足',
                'message': '歩行分析には最低1秒間（約30フレーム）の動画が必要です。'
            }
        
        # 1. 膝角度の時系列データ抽出（視認性付き）
        knee_angles = []
        knee_visibilities = []
        
        for lm in landmarks_history:
            try:
                hip = [lm[24].x, lm[24].y]
                knee = [lm[26].x, lm[26].y]
                ankle = [lm[28].x, lm[28].y]
                
                # Visibility取得
                hip_vis = lm[24].visibility
                knee_vis = lm[26].visibility
                ankle_vis = lm[28].visibility
                avg_vis = (hip_vis + knee_vis + ankle_vis) / 3
                
                # 最低限のVisibilityチェック（0.5以上）
                if avg_vis < 0.5:
                    continue
                    
                angle = self._calculate_angle(hip, knee, ankle)
                knee_angles.append(angle)
                knee_visibilities.append(avg_vis)
            except:
                continue
        
        if len(knee_angles) < 30:
            return {
                'error': 'ランドマーク検出不足',
                'message': '膝・股関節・足首のランドマークが十分に検出できませんでした。'
            }
        
        # 信頼度フィルタを適用
        filtered_knee_angles = self._filter_by_confidence(knee_angles, knee_visibilities)
        
        # 2. 歩行周期の検出
        gait_data = self._detect_gait_cycles(filtered_knee_angles)
        
        if not gait_data or len(gait_data['cycles']) == 0:
            # フォールバック: 単純な最大値評価
            max_extension = max(filtered_knee_angles)
            return {
                'max_knee_angle': round(max_extension, 1),
                'analysis_type': 'simple',
                'recommendations': [
                    "⚠️ 歩行周期が検出できませんでした。",
                    "より長い距離をリラックスして歩いている動画で再測定してください。",
                    f"参考値: 最大膝伸展角度 {round(max_extension, 1)}度"
                ],
                'metrics': {
                    'knee_flexion': round(max_extension, 1)
                }
            }
        
        # 3. 立脚期メトリクスの計算
        knee_metrics = self._calculate_stance_phase_metrics(gait_data)
        
        # 4. 体幹アライメントの評価（修正版）
        trunk_metrics = self._calculate_trunk_alignment(landmarks_history)
        
        # 5. 臨床的フィードバック生成
        recommendations, risk_level = self._generate_clinical_recommendations(
            knee_metrics, trunk_metrics, gait_data
        )
        
        # 6. 結果の統合
        return {
            'analysis_type': 'advanced',
            'gait_cycles_detected': len(gait_data['cycles']),
            'knee_metrics': {
                'mean_stance_extension': round(knee_metrics['mean_stance_extension'], 1),
                'mean_peak_extension': round(knee_metrics['mean_peak_extension'], 1),
                'consistency': round(knee_metrics['consistency'], 1),
                'max_knee_angle': round(knee_metrics['mean_peak_extension'], 1)
            },
            'trunk_metrics': {
                'mean_trunk_angle': round(trunk_metrics['mean_trunk_angle'], 1) if trunk_metrics else None,
                'trunk_variability': round(trunk_metrics['trunk_variability'], 1) if trunk_metrics else None
            } if trunk_metrics else None,
            'risk_level': risk_level,
            'recommendations': recommendations,
            'raw_data': {
                'knee_angles_series': filtered_knee_angles.tolist(),
                'smoothed_angles': gait_data['smoothed_angles'].tolist(),
                'peaks': gait_data['peaks'].tolist(),
                'troughs': gait_data['troughs'].tolist()
            }
        }

    def export_for_sakane_model(self, analysis_result):
        """Sakane 2025モデル用の変数セットをエクスポート"""
        if analysis_result.get('analysis_type') != 'advanced':
            return None
        
        return {
            'variable_1_knee_extension': analysis_result['knee_metrics']['mean_stance_extension'],
            'variable_2_trunk_alignment': analysis_result['trunk_metrics']['mean_trunk_angle'] if analysis_result['trunk_metrics'] else None,
            'variable_3_gait_consistency': analysis_result['knee_metrics']['consistency'],
            'variable_4_step_length': None,
            'variable_5_cadence': None,
            'analysis_timestamp': np.datetime64('now'),
            'model_version': 'Sakane2025_v1.1_confidence_filter'
        }
