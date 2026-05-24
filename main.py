import os
import sys
import yaml
import cv2
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv_module import VideoSampler, StudentDetector, CSVSaver

BEHAVIOR_NAMES = {
    'focus_listen': 'Focus Listening',
    'study_bow': 'Studying (Head Down)',
    'empty_mind': 'Daydreaming',
    'sleep_stu': 'Sleeping',
    'look_side': 'Looking Sideways',
    'talk_discuss': 'Group Discussion',
    'talk_private': 'Private Chat',
    'stand_up': 'Standing',
    'loose_stu': 'Distracted',
    'phone_game': 'Using Phone'
}

def load_config(config_path='config.yaml'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {
        'sampling': {'frame_interval_sec': 3, 'max_frames': 900},
        'detection': {'model': 'yolov8n.pt', 'conf_threshold': 0.3},
        'head_pose': {'pitch_up': 15, 'pitch_down': -10, 'yaw_side': 20},
        'teacher_detection': {
            'podium_bottom_ratio': 0.3,
            'podium_center_width': 0.4,
            'min_standing_height': 0.35,
            'confidence_threshold': 0.7
        },
        'output': {'cache_csv': 'cache_csv/'}
    }

def get_video_files(input_dir):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created input directory: {input_dir}")
        return []
    videos = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    return videos

def validate_frame_data(result):
    """
    校验帧数据：确保行为分类总和不超过检测人数
    返回：(是否有效, 错误信息列表)
    """
    errors = []
    total_stu = result.get('total_stu', result.get('total_students', 0))
    
    behavior_cols = ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
                     'look_side', 'talk_discuss', 'talk_private', 'stand_up',
                     'loose_stu', 'phone_game']
    
    # 计算所有行为分类的总和
    behavior_sum = sum(result.get(col, 0) for col in behavior_cols)
    
    # 校验：行为分类总和不能超过检测人数
    if behavior_sum > total_stu:
        errors.append(f"行为分类总和({behavior_sum})超过检测人数({total_stu})")
        # 修正：按比例缩减各行为分类
        if total_stu > 0:
            ratio = total_stu / behavior_sum
            for col in behavior_cols:
                result[col] = int(result.get(col, 0) * ratio)
            errors.append("已自动修正行为分类数据")
        else:
            # 如果检测人数为0，清空所有行为分类
            for col in behavior_cols:
                result[col] = 0
            errors.append("已清空行为分类数据")
    
    # 再次验证
    behavior_sum = sum(result.get(col, 0) for col in behavior_cols)
    is_valid = behavior_sum <= total_stu
    
    return is_valid, errors

def process_video(video_path, config):
    print(f"\nProcessing: {video_path}")

    sampler = VideoSampler(config_path='config.yaml')
    detector = StudentDetector(config)
    saver = CSVSaver(output_dir=config['output']['cache_csv'])

    try:
        video_info = sampler.get_video_info(video_path)
        if video_info:
            print(f"Video info: {video_info['duration']:.1f}s, {video_info['fps']:.1f} FPS")

        frames = sampler.get_sample_frames(video_path)
        print(f"Sampled {len(frames)} frames")

        if len(frames) == 0:
            print("No frames sampled!")
            return None

        results = []
        validation_errors = []
        
        for i, frame_info in enumerate(frames):
            if (i + 1) % 10 == 0:
                print(f"Processing frame {i + 1}/{len(frames)}...")

            # 检查帧是否有效
            is_valid = frame_info.get('is_valid', True)
            invalid_reason = frame_info.get('invalid_reason', 'valid')
            
            if not is_valid:
                # 无效帧，填充默认数据
                result = {
                    'timestamp': frame_info['timestamp'],
                    'frame_num': frame_info['frame_num'],
                    'is_valid': False,
                    'invalid_reason': invalid_reason,
                    'total_stu': 0,
                    'focus_listen': 0, 'study_bow': 0, 'empty_mind': 0,
                    'sleep_stu': 0, 'look_side': 0, 'talk_discuss': 0,
                    'talk_private': 0, 'stand_up': 0, 'loose_stu': 0, 'phone_game': 0
                }
            else:
                result = detector.process_frame(
                    frame_info['frame'],
                    frame_info['frame_num'],
                    frame_info['timestamp']
                )
                result['is_valid'] = True
                result['invalid_reason'] = 'valid'
                
                # 统一字段名：将 total_students 转换为 total_stu
                if 'total_students' in result and 'total_stu' not in result:
                    result['total_stu'] = result['total_students']
                
                # 数据校验
                _, errors = validate_frame_data(result)
                if errors:
                    validation_errors.append((frame_info['frame_num'], errors))

            results.append(result)

        if validation_errors:
            print("\n⚠️ 数据校验警告:")
            for frame_num, errors in validation_errors[:5]:  # 只显示前5个警告
                print(f"  Frame {frame_num}: {'; '.join(errors)}")
            if len(validation_errors) > 5:
                print(f"  ... 还有 {len(validation_errors) - 5} 个警告")

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = saver.save_frame_data(results, video_name)

        print_summary(results)

        return output_path

    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_summary(results):
    if not results:
        return

    total_frames = len(results)
    total_students = sum(r.get('total_stu', r.get('total_students', 0)) for r in results)
    
    # 统计有效帧和无效帧
    valid_frames = sum(1 for r in results if r.get('is_valid', True))
    invalid_frames = total_frames - valid_frames

    print("\n" + "=" * 50)
    print("  Processing Summary")
    print("=" * 50)
    print(f"Total frames processed: {total_frames}")
    print(f"  Valid frames: {valid_frames}")
    print(f"  Invalid frames: {invalid_frames}")
    print(f"Total student detections: {total_students}")
    print(f"Average students per frame: {total_students / total_frames:.1f}")

    # 只统计行为分类字段
    behavior_cols = ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
                     'look_side', 'talk_discuss', 'talk_private', 'stand_up',
                     'loose_stu', 'phone_game']
    
    for col in behavior_cols:
        count = sum(r.get(col, 0) for r in results)
        ratio = count / total_students * 100 if total_students > 0 else 0
        name = BEHAVIOR_NAMES.get(col, col)
        print(f"  {name}: {count} ({ratio:.1f}%)")

def main():
    print("=" * 60)
    print("  Classroom Attention Analysis System - CV Module")
    print("  Computer Vision Data Extraction")
    print("=" * 60)

    config = load_config()
    input_dir = config.get('video', {}).get('input_dir', 'inputs/')

    videos = get_video_files(input_dir)

    if not videos:
        print(f"\nNo videos found in '{input_dir}' directory.")
        print("Please add video files (.mp4, .avi, .mov) to the inputs folder.")
        print("Example: inputs/classroom.mp4")
        return

    print(f"\nFound {len(videos)} video(s):")
    for v in videos:
        print(f"  - {v}")

    for video_file in videos:
        video_path = os.path.join(input_dir, video_file)
        process_video(video_path, config)

    print("\n" + "=" * 60)
    print("  All videos processed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
