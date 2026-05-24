import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv
import torch


class StudentDetector:
    BEHAVIOR_COLS = [
        'focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
        'look_side', 'talk_discuss', 'talk_private',
        'stand_up', 'loose_stu', 'phone_game'
    ]

    def __init__(self, config=None):
        self.config = config or self._default_config()
        
        # 检测是否有GPU可用
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 使用设备: {self.device}")
        
        self.model = YOLO(self.config['detection']['model'])
        if self.device == 'cuda':
            self.model.to('cuda')
        
        self.frame_count = 0
        self.detection_history = []
        print(f"✅ 检测器初始化完成，使用模型: {self.config['detection']['model']}")

    def _default_config(self):
        return {
            'detection': {
                'model': 'yolov8l.pt',
                'conf_threshold': 0.15,
                'iou_threshold': 0.45,
                'min_bbox_area_ratio': 0.0015,
                'max_bbox_area_ratio': 0.4
            },
            'head_pose': {
                'pitch_up': 15,
                'pitch_down': -10,
                'yaw_side': 20
            }
        }

    def detect(self, frame):
        """
        单帧检测，返回所有检测到的人
        每次调用都是独立的，不受之前帧影响
        """
        results = self.model(
            frame,
            conf=self.config['detection']['conf_threshold'],
            iou=self.config['detection']['iou_threshold'],
            imgsz=1280,
            max_det=100,
            classes=[0],
            verbose=False,
        )[0]

        persons = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                if int(box.cls) == 0:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    persons.append({
                        'bbox': tuple(xyxy),
                        'confidence': conf
                    })
        return persons

    def _filter_by_size(self, persons, frame_shape):
        """根据检测框大小过滤"""
        if len(persons) == 0:
            return persons

        h, w = frame_shape[:2]
        min_area = h * w * self.config['detection']['min_bbox_area_ratio']
        max_area = h * w * self.config['detection']['max_bbox_area_ratio']

        filtered = []
        for p in persons:
            x1, y1, x2, y2 = p['bbox']
            area = (x2 - x1) * (y2 - y1)
            if min_area < area < max_area:
                filtered.append(p)
        return filtered

    def _sort_by_spatial_position(self, persons, frame_shape):
        """
        按空间位置排序（从左到右，从上到下）
        并分配座位号
        """
        if len(persons) == 0:
            return persons

        h, w = frame_shape[:2]

        persons.sort(key=lambda p: (p['bbox'][1], p['bbox'][0]))

        avg_height = np.mean([p['bbox'][3] - p['bbox'][1] for p in persons])
        rows = []
        current_row = [persons[0]]

        for p in persons[1:]:
            if p['bbox'][1] - current_row[-1]['bbox'][1] < avg_height * 0.5:
                current_row.append(p)
            else:
                rows.append(current_row)
                current_row = [p]
        rows.append(current_row)

        sorted_persons = []
        seat_num = 1
        for row in rows:
            row.sort(key=lambda p: p['bbox'][0])
            for p in row:
                p['seat_id'] = seat_num
                sorted_persons.append(p)
                seat_num += 1
        return sorted_persons

    def _classify_behavior(self, bbox, frame_shape):
        """
        根据检测框特征分类行为
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        person_h = y2 - y1
        person_w = x2 - x1
        person_area = person_h * person_w
        frame_area = h * w
        aspect_ratio = person_w / person_h if person_h > 0 else 1

        if aspect_ratio > 0.9 and y2 > h * 0.7 and person_h < h * 0.18:
            return 'sleep_stu'

        if aspect_ratio < 0.4:
            return 'look_side'

        if person_h > h * 0.35 and y1 < h * 0.35:
            return 'stand_up'

        if 0.008 < person_area / frame_area < 0.04 and y1 > h * 0.2:
            return 'study_bow'

        if 0.003 < person_area / frame_area < 0.015:
            return 'empty_mind'

        return 'focus_listen'

    def process_frame(self, frame, frame_num, timestamp):
        """
        处理单帧：检测 + 空间位置排序 + 行为分类
        无跟踪器，每帧独立检测
        """
        self.frame_count += 1
        h, w = frame.shape[:2]

        persons = self.detect(frame)

        persons = self._filter_by_size(persons, frame.shape)

        persons = self._sort_by_spatial_position(persons, frame.shape)

        behavior_counts = {col: 0 for col in self.BEHAVIOR_COLS}
        frame_students = []

        for p in persons:
            bbox = p['bbox']
            behavior = self._classify_behavior(bbox, (h, w))
            behavior_counts[behavior] += 1

            frame_students.append({
                'tracker_id': p.get('seat_id', 0),
                'bbox': bbox,
                'center': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                'behavior': behavior,
                'confidence': p['confidence'],
                'is_teacher': False,
                'pitch': 0,
                'yaw': 0
            })

        return {
            'frame_num': frame_num,
            'timestamp': timestamp,
            'total_stu': len(persons),
            'total_teachers': 0,
            'behavior_counts': behavior_counts,
            'students': frame_students,
            'teachers': [],
            'is_valid': True
        }