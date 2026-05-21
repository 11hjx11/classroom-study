"""
课堂专注度分析系统 - 可视化演示版（含多目标跟踪）
功能：实时显示检测框、跟踪ID、行为标签、运动轨迹
"""

import cv2
import os
import sys
import yaml
from datetime import datetime

sys.path.append('.')

try:
    from src.cv_module.student_detector import StudentDetector
except ImportError as e:
    print(f"Error: Cannot import StudentDetector - {e}")
    sys.exit(1)

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except:
    config = {
        'detection': {'model': 'yolov8n.pt', 'conf_threshold': 0.3}
    }

BEHAVIOR_COLORS = {
    'focus_listen': (0, 255, 0),
    'study_bow': (255, 255, 0),
    'empty_mind': (0, 165, 255),
    'sleep_stu': (0, 0, 255),
    'look_side': (255, 0, 255),
    'talk_discuss': (255, 255, 255),
    'talk_private': (0, 255, 255),
    'stand_up': (255, 0, 0),
    'loose_stu': (128, 128, 128),
    'phone_game': (0, 0, 128)
}

BEHAVIOR_NAMES = {
    'focus_listen': 'Focus',
    'study_bow': 'Study',
    'empty_mind': 'Empty',
    'sleep_stu': 'Sleep',
    'look_side': 'Look',
    'talk_discuss': 'Talk',
    'talk_private': 'Private',
    'stand_up': 'Stand',
    'loose_stu': 'Loose',
    'phone_game': 'Phone'
}


def draw_statistics(frame, behavior_counts, total_students):
    panel_w = 260
    panel_h = 200

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"Students: {total_students}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    y = 55
    for behavior, name in BEHAVIOR_NAMES.items():
        count = behavior_counts.get(behavior, 0)
        if count > 0:
            color = BEHAVIOR_COLORS.get(behavior, (255, 255, 255))
            cv2.putText(frame, f"{name}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 22

    return frame


def main():
    print("=" * 60)
    print("课堂专注度分析系统 - 可视化演示（ByteTrack多目标跟踪）")
    print("=" * 60)

    inputs_dir = 'inputs'
    if not os.path.exists(inputs_dir):
        os.makedirs(inputs_dir)
        print(f"请将视频放入 {inputs_dir}/ 文件夹")
        return

    videos = [f for f in os.listdir(inputs_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not videos:
        print("❌ 未找到视频文件")
        print(f"请将课堂视频放入 {inputs_dir}/ 文件夹")
        return

    print(f"📁 视频: {videos[0]}")
    video_path = os.path.join(inputs_dir, videos[0])

    print("初始化检测器（含ByteTrack）...")
    detector = StudentDetector(config)
    print("✅ 检测器初始化完成")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    print(f"FPS: {fps:.1f}, 总帧数: {total_frames}")
    print("\n按 'q' 退出, 's' 截图, 't' 显示/隐藏轨迹\n")

    students = []
    behavior_counts = {}
    total_stu = 0
    show_trajectory = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 3 == 0:
            result = detector.process_frame(frame, frame_count, frame_count / fps)
            behavior_counts = {k: v for k, v in result.items() if k in BEHAVIOR_NAMES}
            students = result.get('students', [])
            total_stu = result.get('total_stu', 0)

        if show_trajectory:
            trajectories = detector.get_trajectories()
            for tid, traj in trajectories.items():
                if len(traj) > 1:
                    color = ((tid * 50) % 255, (tid * 100) % 255, (tid * 150) % 255)
                    for i in range(1, len(traj)):
                        cv2.line(frame, traj[i-1], traj[i], color, 2)

        for student in students:
            x1, y1, x2, y2 = student['bbox']
            behavior = student['behavior']
            tracker_id = student['tracker_id']
            color = BEHAVIOR_COLORS.get(behavior, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{tracker_id}|{BEHAVIOR_NAMES.get(behavior, behavior)}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 8), (x1 + label_w + 6, y1 - 2), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        frame = draw_statistics(frame, behavior_counts, total_stu)

        progress = frame_count / total_frames * 100
        cv2.putText(frame, f"Progress: {progress:.1f}%", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, "q:quit s:screenshot t:trajectory", (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Classroom Study - ByteTrack Tracking', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs('screenshots', exist_ok=True)
            path = f"screenshots/frame_{frame_count}_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(path, frame)
            print(f"📸 截图: {path}")
        elif key == ord('t'):
            show_trajectory = not show_trajectory
            print(f"轨迹显示: {'开启' if show_trajectory else '关闭'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ 演示完成")


if __name__ == "__main__":
    main()