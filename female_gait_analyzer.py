import numpy as np

class FemaleGaitAnalyzer:
    def __init__(self):
        # 臨床的な閾値
        self.thresholds = {
            'knee_extension_ideal': 175.0,
            'risk_threshold': 165.0
        }

    def _calculate_angle(self, a, b, c):
        """3点の座標から角度を計算（180度を最大とする）"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def analyze_clinical_data(self, landmarks_history):
        if not landmarks_history:
            return None

        knee_angles = []
        for lm in landmarks_history:
            # 股関節(24), 膝(26), 足首(28)
            # lm[24].visibility などで精度チェックを入れるのもあり
            hip = [lm[24].x, lm[24].y]
            knee = [lm[26].x, lm[26].y]
            ankle = [lm[28].x, lm[28].y]
            knee_angles.append(self._calculate_angle(hip, knee, ankle))

        max_extension = max(knee_angles) if knee_angles else 0
        
        # --- すみれん流：臨床アドバイス生成 ---
        recs = []
        if max_extension < self.thresholds['risk_threshold']:
            # 「過度のプロセス（背景の推測）」と「共感」を注入
            recs.append("✨ **膝の伸びに隠された『頑張り』への気づき**")
            recs.append(f"現在の最大伸展は{round(max_extension, 1)}度です。膝が伸びきる手前で止まっているのは、もしかすると長年の歩行習慣や、無意識に膝を保護しようとする『優しさ』の結果かもしれません。")
            recs.append("💡 **理学療法士からの処方箋**：無理に伸ばそうとせず、まずは椅子に座って膝の裏をゆっくり床に近づける『等尺性収縮』から始めてみませんか？働くあなたの身体を、少しずつ解放していきましょう。")
        else:
            recs.append("✅ **しなやかで力強い膝の伸びです**")
            recs.append(f"最大{round(max_extension, 1)}度までしっかり伸びています。これは骨盤底筋や体幹が安定している証拠でもありますね。")
            recs.append("🚀 **さらなる生産性向上へ**：この歩き方を維持することで、夕方の疲れにくさが変わってきます。素晴らしい状態です！")

        return {
            'max_knee_angle': round(max_extension, 1),
            'recommendations': recs,
            # 将来的にSakane 2025の多変数を入れるための拡張枠
            'metrics': {
                'knee_flexion': round(max_extension, 1),
                'symmetry_score': 85 # 仮
            }
        }
