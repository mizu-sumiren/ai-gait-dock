import cv2
import mediapipe as mp

class MediaPipeAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                data.append({
                    'left_ankle': [landmarks[27].x, landmarks[27].y],
                    'right_ankle': [landmarks[28].x, landmarks[28].y],
                    'left_knee': [landmarks[25].x, landmarks[25].y],
                    'left_hip': [landmarks[23].x, landmarks[23].y]
                })
        cap.release()
        return data