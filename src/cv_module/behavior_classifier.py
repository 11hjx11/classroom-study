import cv2
import mediapipe as mp
import numpy as np

class BehaviorClassifier:
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            min_detection_confidence=0.5
        )

    def _default_config(self):
        return {
            'head_pose': {
                'pitch_up': 15,
                'pitch_down': -10,
                'yaw_side': 20
            }
        }

    def classify_student(self, person_bbox, frame, head_pose_detector=None):
        x1, y1, x2, y2 = map(int, person_bbox)
        h, w = frame.shape[:2]
        person_h = y2 - y1
        person_w = x2 - x1

        if self._is_sleeping(person_bbox, h):
            return 'sleep_stu'

        if self._is_standing(person_bbox, h):
            return 'stand_up'

        margin = 20
        fx1 = max(0, x1 - margin)
        fy1 = max(0, y1 - margin)
        fx2 = min(w, x2 + margin)
        fy2 = min(h, y2 + margin)
        face_roi = frame[fy1:fy2, fx1:fx2]

        if head_pose_detector is not None:
            pitch, yaw, head_state = head_pose_detector.detect(
                face_roi,
                self.config['head_pose']
            )

            if head_state is not None:
                if head_state == 'focus':
                    return 'focus_listen'
                elif head_state == 'look_up':
                    return 'stand_up' if y1 < h * 0.3 else 'look_side'
                elif head_state == 'look_down':
                    if person_w < person_h * 0.6:
                        return 'phone_game'
                    else:
                        return 'study_bow'
                elif head_state == 'look_side':
                    return 'look_side'

        aspect_ratio = person_w / person_h if person_h > 0 else 1

        if aspect_ratio > 0.9:
            if y2 > h * 0.8:
                return 'sleep_stu'
            else:
                return 'study_bow'
        elif aspect_ratio < 0.5:
            if y1 < h * 0.3:
                return 'stand_up'
            else:
                return 'look_side'
        else:
            return 'focus_listen'

    def _is_sleeping(self, bbox, frame_height):
        x1, y1, x2, y2 = bbox
        person_h = y2 - y1
        person_w = x2 - x1
        return person_w > person_h * 1.2 and y2 > frame_height * 0.7

    def _is_standing(self, bbox, frame_height):
        y1 = bbox[1]
        return y1 < frame_height * 0.3

    def classify_group_behavior(self, detections, frame):
        if len(detections) < 4:
            return None
        return 'talk_private'
