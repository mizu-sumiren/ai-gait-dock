import numpy as np

class FemaleGaitAnalyzer:
    def __init__(self):
        # 臨床的な閾値（理学療法士の知見）
        self.thresholds = {
            'knee_extension_ideal': 175.0, # ほぼ真っ直ぐ
            'knee_flexion_swing_min': 60.0 # 遊脚期のクリアランス
        }

    def _calculate_angle(self, a, b, c):
        """3点の座標(x, y)から角度を計算する"""
        a = np.array(a) # 股関節
        b = np.array(b) # 膝
        c = np.array(c) # 足首

        # ベクトルを生成
        ba = a - b
        bc = c - b

        # 内積から角度(ラジアン)を算出
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 0

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def analyze_clinical_data(self, landmarks_history):
        """
        landmarks_history: フレームごとの骨格データリスト
        """
        knee_angles = []
        
        if not landmarks_history:
            return None

        for landmarks in landmarks_history:
            # 右側の膝角度を計算（横向き動画想定）
            # MediaPipe Index: Hip(24), Knee(26), Ankle(28)
            # 座標が取得できているかチェック
            if landmarks[24] and landmarks[26] and landmarks[28]:
                hip = [landmarks[24].x, landmarks[24].y]
                knee = [landmarks[26].x, landmarks[26].y]
                ankle = [landmarks[28].x, landmarks[28].y]
                
                angle = self._calculate_angle(hip, knee, ankle)
                knee_angles.append(angle)

        # 最大伸展位（最も膝が伸びた角度）を抽出
        # 通常、歩行周期の中で最も180度に近い値
        max_extension = max(knee_angles) if knee_angles else 0
        
        # --- すみれん流：臨床アドバイスロジック ---
        risk_score = 0
        recs = []
        
        if max_extension < 165:
            risk_score = 40
            recs.append("✨ **膝の伸びへの気づき**")
            recs.append("歩行の中で膝が伸びきる手前で止まっているようです。これは膝への負担を減らそうとする、体なりの工夫かもしれません。")
            recs.append("💡 **セルフケアのヒント**：まずは「お皿の周り」を優しくさすってほぐすことから始めましょう。それだけで、足がスッと前に出やすくなる感覚が得られるはずです。")
        else:
            risk_score = 10
            recs.append("✅ **素晴らしい膝の伸びです**")
            recs.append("膝がしっかり伸びることで、地面を効率よく蹴り出せています。働く人の力強い歩き方ですね！")

        return {
            'max_knee_angle': round(max_extension, 1),
            'risk_score': risk_score,
            'recommendations': recs
        }
