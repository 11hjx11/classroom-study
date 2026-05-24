import cv2
import yaml
import os

class VideoSampler:
    def __init__(self, config_path='config.yaml'):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {
            'sampling': {
                'frame_interval_sec': 3,
                'max_frames': 1000
            }
        }

    def get_sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        base_interval_sec = self.config['sampling'].get('frame_interval_sec', 3)
        max_frames = self.config['sampling'].get('max_frames', 1000)
        
        duration_minutes = duration / 60
        
        if duration_minutes <= 20:
            interval_sec = 3
        elif duration_minutes <= 40:
            interval_sec = 4
        else:
            interval_sec = 5
        
        interval = int(interval_sec * fps)

        frames_info = []
        frame_num = 0
        sampled = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % interval == 0:
                timestamp = frame_num / fps if fps > 0 else 0
                frames_info.append({
                    'frame_num': frame_num,
                    'timestamp': round(timestamp, 2),
                    'frame': frame,
                    'duration': duration
                })
                sampled += 1

                if sampled >= max_frames:
                    break

            frame_num += 1

        cap.release()
        print(f"Sampled {len(frames_info)} frames from video")
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
