import cv2
import yaml
import os
import numpy as np

class VideoSampler:
    def __init__(self, config_path='config.yaml'):
        self.config = self._load_config(config_path)

    def _is_valid_frame(self, frame):
        """
        判断帧是否为有效帧
        无效帧包括：黑屏、模糊、无学生画面等
        """
        if frame is None or frame.size == 0:
            return False, "empty frame"

        # 检查是否为黑屏（像素值接近0）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 10:
            return False, "black screen"

        # 检查是否过度曝光（像素值接近255）
        if mean_brightness > 245:
            return False, "overexposed"

        # 检查图像是否模糊（使用拉普拉斯方差）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian)
        if blur_score < 50:
            return False, "blurry"

        # 检查是否有足够的内容（非纯色画面）
        std_dev = np.std(gray)
        if std_dev < 5:
            return False, "uniform color"

        return True, "valid"

    def _load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return self._default_config()

    def _default_config(self):
        return {
            'sampling': {
                'frame_interval_sec': 5,
                'max_frames': 500
            }
        }

    def get_sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        interval = self.config['sampling']['frame_interval_sec'] * fps
        max_frames = self.config['sampling']['max_frames']

        frames_info = []
        frame_num = 0
        sampled = 0
        invalid_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % int(interval) == 0:
                timestamp = frame_num / fps if fps > 0 else 0
                
                # 检测帧有效性
                is_valid, invalid_reason = self._is_valid_frame(frame)
                
                frames_info.append({
                    'frame_num': frame_num,
                    'timestamp': round(timestamp, 2),
                    'frame': frame,
                    'duration': duration,
                    'is_valid': is_valid,
                    'invalid_reason': invalid_reason
                })
                
                if is_valid:
                    sampled += 1
                else:
                    invalid_count += 1

                if sampled >= max_frames:
                    break

            frame_num += 1

        cap.release()
        print(f"Sampled {len(frames_info)} frames total, {sampled} valid, {invalid_count} invalid")
        return frames_info

    def get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'width': width,
            'height': height
        }
