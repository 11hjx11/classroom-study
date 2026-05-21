"""
仿真数据生成器
用于生成模拟课堂行为数据，测试数据分析模块
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


class SimulationDataGenerator:
    """生成仿真课堂行为数据"""
    
    def __init__(self, output_dir='cache_csv/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 行为列定义
        self.behavior_cols = [
            'focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
            'look_side', 'talk_discuss', 'talk_private',
            'stand_up', 'loose_stu', 'phone_game'
        ]
        
        # 时段定义（45分钟课程，5秒一帧 = 540帧）
        self.segments = {
            'opening': {'name': '开课适应期', 'start_frame': 0, 'end_frame': 108, 'duration': 9},
            'efficient': {'name': '高效学习期', 'start_frame': 108, 'end_frame': 324, 'duration': 18},
            'fatigue': {'name': '疲劳下滑期', 'start_frame': 324, 'end_frame': 432, 'duration': 9},
            'closing': {'name': '下课涣散期', 'start_frame': 432, 'end_frame': 540, 'duration': 9}
        }
    
    def _get_segment(self, frame_num):
        """获取当前帧所属时段"""
        for seg_id, seg_info in self.segments.items():
            if seg_info['start_frame'] <= frame_num < seg_info['end_frame']:
                return seg_id
        return 'closing'
    
    def _generate_frame_data(self, frame_num, total_frames, avg_students=30):
        """生成单帧数据"""
        timestamp = frame_num * 5  # 5秒一帧
        segment = self._get_segment(frame_num)
        
        # 根据时段设置行为分布
        segment_params = {
            'opening': {
                'focus_listen': 0.4, 'study_bow': 0.15, 'empty_mind': 0.2,
                'sleep_stu': 0.02, 'look_side': 0.15, 'talk_discuss': 0.03,
                'talk_private': 0.03, 'stand_up': 0.01, 'loose_stu': 0.01, 'phone_game': 0.0
            },
            'efficient': {
                'focus_listen': 0.6, 'study_bow': 0.25, 'empty_mind': 0.05,
                'sleep_stu': 0.01, 'look_side': 0.04, 'talk_discuss': 0.03,
                'talk_private': 0.01, 'stand_up': 0.0, 'loose_stu': 0.0, 'phone_game': 0.01
            },
            'fatigue': {
                'focus_listen': 0.4, 'study_bow': 0.15, 'empty_mind': 0.15,
                'sleep_stu': 0.08, 'look_side': 0.15, 'talk_discuss': 0.02,
                'talk_private': 0.03, 'stand_up': 0.01, 'loose_stu': 0.01, 'phone_game': 0.0
            },
            'closing': {
                'focus_listen': 0.2, 'study_bow': 0.1, 'empty_mind': 0.25,
                'sleep_stu': 0.1, 'look_side': 0.2, 'talk_discuss': 0.02,
                'talk_private': 0.1, 'stand_up': 0.02, 'loose_stu': 0.01, 'phone_game': 0.0
            }
        }
        
        params = segment_params[segment]
        
        # 添加随机波动
        for key in params:
            params[key] += np.random.uniform(-0.05, 0.05)
            params[key] = max(0, min(1, params[key]))
        
        # 归一化概率
        total_prob = sum(params.values())
        for key in params:
            params[key] /= total_prob
        
        # 生成人数（添加随机噪声）
        current_students = max(25, int(avg_students + np.random.normal(0, 2)))
        total_stu = current_students
        
        # 根据概率分布生成各行为人数
        behavior_counts = {}
        remaining = current_students
        
        for col in self.behavior_cols[:-1]:
            count = int(params[col] * current_students + np.random.normal(0, 0.5))
            count = max(0, min(remaining, count))
            behavior_counts[col] = count
            remaining -= count
        
        behavior_counts[self.behavior_cols[-1]] = remaining
        
        # 确保总和正确
        while sum(behavior_counts.values()) != current_students:
            diff = current_students - sum(behavior_counts.values())
            if diff > 0:
                behavior_counts['focus_listen'] += diff
            else:
                behavior_counts['focus_listen'] += diff
        
        # 生成有效帧标志（随机生成一些无效帧）
        is_valid = True
        invalid_reason = ''
        
        # 随机生成无效帧（约5%的概率）
        if np.random.random() < 0.05:
            is_valid = False
            reasons = ['black_screen', 'blur', 'over_exposed', 'motion_blur']
            invalid_reason = reasons[np.random.randint(len(reasons))]
            # 无效帧设置所有行为为0
            for col in self.behavior_cols:
                behavior_counts[col] = 0
            total_stu = 0
        
        return {
            'timestamp': timestamp,
            'frame_num': frame_num,
            'is_valid': is_valid,
            'invalid_reason': invalid_reason,
            'total_stu': total_stu,
            **behavior_counts
        }
    
    def generate_data(self, total_frames=540, avg_students=30, video_name='demo'):
        """生成完整的仿真数据"""
        frames_data = []
        
        for frame_num in range(total_frames):
            frame_data = self._generate_frame_data(frame_num, total_frames, avg_students)
            frames_data.append(frame_data)
        
        # 创建DataFrame
        df = pd.DataFrame(frames_data)
        
        # 调整列顺序
        columns = [
            'timestamp', 'frame_num', 'is_valid', 'invalid_reason', 'total_stu',
            'focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
            'look_side', 'talk_discuss', 'talk_private',
            'stand_up', 'loose_stu', 'phone_game'
        ]
        df = df[columns]
        
        # 保存CSV
        output_path = f"{self.output_dir}/{video_name}_raw.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 仿真数据生成完成！")
        print(f"   文件: {output_path}")
        print(f"   帧数: {len(df)}")
        print(f"   有效帧: {df['is_valid'].sum()}")
        print(f"   无效帧: {len(df) - df['is_valid'].sum()}")
        print(f"   平均检测人数: {df['total_stu'].mean():.1f}")
        
        return df
    
    def generate_multiple_datasets(self, count=5):
        """生成多个仿真数据集"""
        for i in range(count):
            video_name = f"classroom_{i+1}"
            # 每个班级有不同的平均人数
            avg_students = np.random.randint(25, 35)
            self.generate_data(avg_students=avg_students, video_name=video_name)


if __name__ == '__main__':
    generator = SimulationDataGenerator()
    
    # 生成单个数据集
    print("=" * 60)
    print("生成单个仿真数据集...")
    print("=" * 60)
    df = generator.generate_data()
    
    # 生成多个数据集
    print("\n" + "=" * 60)
    print("生成多个仿真数据集...")
    print("=" * 60)
    generator.generate_multiple_datasets(count=3)