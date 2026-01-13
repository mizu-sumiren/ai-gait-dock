import cv2
import mediapipe as mp
# 取り出し方をより確実に変更
from mediapipe.python.solutions import pose as mp_pose

class MediaPipeAnalyzer:
    def __init__(self):
        # 直接ポーズ検出機能を読み込む
        self.mp_pose = mp_pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        data = []
        if not cap.isOpened():
            return None
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # 必要な指標（足首、膝、股関節）を抽出
                data.append({
                    'left_ankle': [landmarks[27].x, landmarks[27].y],
                    'right_ankle': [landmarks[28].x, landmarks[28].y],
                    'left_knee': [landmarks[25].x, landmarks[25].y],
                    'left_hip': [landmarks[23].x, landmarks[23].y],
                    'left_shoulder': [landmarks[11].x, landmarks[11].y]
                })
        cap.release()
        return data

    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()
