import cv2
import mediapipe as mp
import numpy as np

class HeadPoseDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.model_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0]
        ], dtype=np.float32)
        self.landmark_indices = [1, 152, 33, 263, 61, 291]

    def compute_head_pose(self, landmarks, img_w, img_h):
        image_points = []
        for idx in self.landmark_indices:
            x = landmarks.landmark[idx].x * img_w
            y = landmarks.landmark[idx].y * img_h
            image_points.append([x, y])
        image_points = np.array(image_points, dtype=np.float32)

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, _ = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0, 0

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)

        if sy < 1e-6:
            pitch = 0
            yaw = 0
        else:
            pitch = np.arctan2(-rot_mat[2, 0], sy) * 180 / np.pi
            yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0]) * 180 / np.pi

        return pitch, yaw

    def classify_attention(self, pitch, yaw, config=None):
        if config is None:
            config = {'pitch_up': 15, 'pitch_down': -10, 'yaw_side': 20}

        if pitch > config['pitch_up']:
            return 'look_up'
        elif pitch < config['pitch_down']:
            return 'look_down'
        elif abs(yaw) > config['yaw_side']:
            return 'look_side'
        else:
            return 'focus'

    def detect(self, face_roi, config=None):
        if face_roi.size == 0:
            return None, None, None

        h, w = face_roi.shape[:2]
        if h < 20 or w < 20:
            return None, None, None

        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_face)

        if not results.multi_face_landmarks:
            return None, None, None

        landmarks = results.multi_face_landmarks[0]
        pitch, yaw = self.compute_head_pose(landmarks, w, h)
        state = self.classify_attention(pitch, yaw, config)

        return pitch, yaw, state
