import cv2
from ultralytics import YOLO
import supervision as sv
from collections import deque, Counter
import numpy as np


class StudentDetector:
    BEHAVIOR_COLS = [
        'focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
        'look_side', 'talk_discuss', 'talk_private',
        'stand_up', 'loose_stu', 'phone_game'
    ]

    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.model = YOLO(self.config['detection']['model'])
        
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=40,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        self.trajectories = {}
        self.state_history = {}
        
        # 教师检测相关
        self.teacher_track_ids = set()
        self.teacher_history = deque(maxlen=10)

    def _default_config(self):
        return {
            'detection': {
                'model': 'yolov8n.pt',
                'conf_threshold': 0.3
            },
            'head_pose': {
                'pitch_up': 15,
                'pitch_down': -10,
                'yaw_side': 20
            },
            'teacher_detection': {
                'podium_bottom_ratio': 0.3,  # 讲台区域占画面底部比例（视频底部）
                'podium_center_width': 0.4,  # 讲台区域宽度（中心两侧各20%）
                'min_standing_height': 0.35, # 站立最小高度比例
                'confidence_threshold': 0.7  # 教师判定置信度阈值
            }
        }

    def _is_in_podium_area(self, bbox, frame_shape):
        """
        判断目标是否在讲台区域（画面底部中央）
        根据用户反馈：视频样本中老师和讲台位于视频底部
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        
        # 讲台区域：底部区域 + 中央区域
        podium_bottom = h * (1 - self.config['teacher_detection']['podium_bottom_ratio'])
        center_left = w * (0.5 - self.config['teacher_detection']['podium_center_width'] / 2)
        center_right = w * (0.5 + self.config['teacher_detection']['podium_center_width'] / 2)
        
        # 目标顶部在讲台区域内（目标位于画面底部）
        return y1 >= podium_bottom and x1 >= center_left and x2 <= center_right

    def _is_standing(self, bbox, frame_shape):
        """
        判断目标是否站立
        通过检测框高度与画面高度的比例判断
        """
        h = frame_shape[0]
        x1, y1, x2, y2 = bbox
        person_height = y2 - y1
        
        return person_height > h * self.config['teacher_detection']['min_standing_height']

    def _detect_teacher(self, detections, frame_shape):
        """
        检测教师
        基于两个特征：1. 在讲台区域 2. 站立姿态
        """
        teacher_indices = []
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else None
            
            # 检查是否在讲台区域且站立
            in_podium = self._is_in_podium_area(bbox, frame_shape)
            is_standing = self._is_standing(bbox, frame_shape)
            
            if in_podium and is_standing:
                # 使用时间平滑来稳定教师判定
                if tracker_id:
                    self.teacher_history.append((tracker_id, True))
                    recent_decisions = [d[1] for d in self.teacher_history if d[0] == tracker_id][-5:]
                    if len(recent_decisions) >= 3 and sum(recent_decisions) >= 3:
                        self.teacher_track_ids.add(tracker_id)
                        teacher_indices.append(i)
            
            elif tracker_id in self.teacher_track_ids:
                # 如果之前判定为教师但现在不在讲台区域，暂时保留身份
                # 允许教师短暂离开讲台
                self.teacher_history.append((tracker_id, False))
                recent_decisions = [d[1] for d in self.teacher_history if d[0] == tracker_id][-5:]
                if len(recent_decisions) >= 5 and sum(recent_decisions) == 0:
                    # 连续5帧不在讲台区域，移除教师身份
                    self.teacher_track_ids.discard(tracker_id)
        
        return teacher_indices

    def detect_students(self, frame):
        results = self.model(frame, conf=self.config['detection']['conf_threshold'])[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]

        if len(detections) > 0:
            detections = self.tracker.update_with_detections(detections)
        else:
            detections = sv.Detections.empty()

        return detections

    def classify_behavior_by_box(self, bbox, frame_shape):
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        person_h = y2 - y1
        person_w = x2 - x1

        if y1 < h * 0.3 and person_h > h * 0.35:
            return 'stand_up'

        aspect_ratio = person_w / person_h if person_h > 0 else 1
        if aspect_ratio > 0.9 and y2 > h * 0.7:
            return 'sleep_stu'

        if aspect_ratio < 0.5:
            return 'look_side'

        return 'focus_listen'

    def process_frame(self, frame, frame_num, timestamp):
        h, w = frame.shape[:2]
        detections = self.detect_students(frame)
        
        # 检测教师
        teacher_indices = self._detect_teacher(detections, frame.shape)
        total_teachers = len(teacher_indices)
        
        # 分离学生和教师检测结果
        student_detections = sv.Detections.empty()
        if len(detections) > 0:
            # 创建学生检测（排除教师）
            student_mask = np.ones(len(detections), dtype=bool)
            student_mask[teacher_indices] = False
            
            if student_mask.any():
                student_detections = sv.Detections(
                    xyxy=detections.xyxy[student_mask],
                    confidence=detections.confidence[student_mask],
                    class_id=detections.class_id[student_mask],
                    tracker_id=detections.tracker_id[student_mask] if detections.tracker_id is not None else None
                )
        
        total_students = len(student_detections)

        behavior_counts = {col: 0 for col in self.BEHAVIOR_COLS}
        frame_students = []
        frame_teachers = []

        # 处理学生
        if student_detections.tracker_id is not None and len(student_detections) > 0:
            for i in range(len(student_detections)):
                xyxy = student_detections.xyxy[i]
                tracker_id = int(student_detections.tracker_id[i])
                confidence = student_detections.confidence[i] if student_detections.confidence is not None else 0.0

                x1, y1, x2, y2 = map(int, xyxy)

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if tracker_id not in self.trajectories:
                    self.trajectories[tracker_id] = deque(maxlen=30)
                self.trajectories[tracker_id].append(center)

                behavior = self.classify_behavior_by_box((x1, y1, x2, y2), frame.shape)

                if tracker_id not in self.state_history:
                    self.state_history[tracker_id] = deque(maxlen=5)
                self.state_history[tracker_id].append(behavior)

                if len(self.state_history[tracker_id]) == 5:
                    stable_behavior = Counter(self.state_history[tracker_id]).most_common(1)[0][0]
                else:
                    stable_behavior = behavior

                if stable_behavior in behavior_counts:
                    behavior_counts[stable_behavior] += 1
                else:
                    behavior_counts['focus_listen'] += 1

                frame_students.append({
                    'tracker_id': tracker_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'behavior': stable_behavior,
                    'confidence': float(confidence),
                    'pitch': 0,
                    'yaw': 0,
                    'is_teacher': False
                })
        
        # 处理教师
        if detections.tracker_id is not None and len(detections) > 0:
            for i in teacher_indices:
                xyxy = detections.xyxy[i]
                tracker_id = int(detections.tracker_id[i])
                confidence = detections.confidence[i] if detections.confidence is not None else 0.0
                
                x1, y1, x2, y2 = map(int, xyxy)
                
                frame_teachers.append({
                    'tracker_id': tracker_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'confidence': float(confidence),
                    'is_teacher': True
                })

        return {
            'timestamp': timestamp,
            'frame_num': frame_num,
            'total_stu': total_students,
            'total_teachers': total_teachers,
            **behavior_counts,
            'students': frame_students,
            'teachers': frame_teachers
        }

    def get_trajectories(self):
        return {tid: list(traj) for tid, traj in self.trajectories.items()}

    def reset_tracker(self):
        self.tracker.reset()
        self.trajectories.clear()
        self.state_history.clear()
        self.teacher_track_ids.clear()
        self.teacher_history.clear()

    def _empty_result(self, timestamp, frame_num):
        return {
            'timestamp': timestamp,
            'frame_num': frame_num,
            'total_stu': 0,
            'total_teachers': 0,
            **{col: 0 for col in self.BEHAVIOR_COLS}
        }