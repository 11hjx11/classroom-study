"""
时间分析模块
依据课堂教学规律划分时段并聚合统计数据
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List


class TemporalAnalyzer:
    """时间分析器"""
    
    def __init__(self):
        self.behavior_columns = [
            'focus_listen', 'study_bow', 'empty_mind', 'sleep_stu', 
            'look_side', 'talk_discuss', 'talk_private',
            'stand_up', 'loose_stu', 'phone_game'
        ]
        
        # 支持的学生数列名
        self.total_students_columns = ['total_students', 'total_stu']
        
        # 时段定义（基于45分钟课堂）
        self.segment_config = {
            'opening': {'name': '开课适应期', 'start': 0, 'end': 9, 'color': '#00d4ff'},
            'efficient': {'name': '高效学习期', 'start': 9, 'end': 27, 'color': '#10b981'},
            'fatigue': {'name': '疲劳下滑期', 'start': 27, 'end': 36, 'color': '#f59e0b'},
            'closing': {'name': '下课涣散期', 'start': 36, 'end': 45, 'color': '#ef4444'}
        }
    
    def _get_total_students_col(self, df):
        """获取学生总数列名"""
        for col in self.total_students_columns:
            if col in df.columns:
                return col
        return 'total_students'
    
    def determine_class_duration(self, df: pd.DataFrame) -> float:
        """
        根据数据帧计算课堂时长（分钟）
        假设采样频率为5秒/帧
        """
        if len(df) == 0:
            return 45.0  # 默认45分钟
        
        # 获取最后一帧的时间戳
        if 'timestamp' in df.columns:
            last_timestamp = df['timestamp'].iloc[-1]
            return last_timestamp / 60.0
        else:
            # 假设5秒一帧
            return (len(df) * 5) / 60.0
    
    def map_frame_to_segment(self, frame_index: int, total_frames: int, class_duration: float) -> str:
        """
        将帧映射到对应的时段
        """
        if total_frames == 0:
            return 'efficient'
        
        # 计算当前帧对应的分钟数
        frame_time_minutes = (frame_index * 5) / 60.0
        
        # 按比例分配时段
        for segment_id, config in self.segment_config.items():
            segment_start = (config['start'] / 45.0) * class_duration
            segment_end = (config['end'] / 45.0) * class_duration
            
            if segment_start <= frame_time_minutes < segment_end:
                return segment_id
        
        return 'closing'
    
    def add_segment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为每帧添加时段标签
        """
        df = df.copy()
        class_duration = self.determine_class_duration(df)
        total_frames = len(df)
        
        df['segment'] = df.apply(
            lambda row: self.map_frame_to_segment(row.name, total_frames, class_duration),
            axis=1
        )
        
        df['segment_name'] = df['segment'].map(
            lambda x: self.segment_config[x]['name']
        )
        
        return df
    
    def segment_statistics(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        分时段聚合统计数据
        """
        segment_stats = {}
        total_col = self._get_total_students_col(df)
        
        for segment_id, config in self.segment_config.items():
            segment_data = df[df['segment'] == segment_id]
            
            if len(segment_data) == 0:
                continue
            
            # 计算各项统计
            stats = {
                'segment_id': segment_id,
                'segment_name': config['name'],
                'frame_count': len(segment_data),
                'avg_total_students': segment_data[total_col].mean().round(1),
                'max_total_students': segment_data[total_col].max(),
                'min_total_students': segment_data[total_col].min()
            }
            
            # 各行为的平均值和占比
            for col in self.behavior_columns:
                if col in df.columns:
                    avg_val = segment_data[col].mean().round(1)
                    stats[f'avg_{col}'] = avg_val
                    
                    total_avg = stats['avg_total_students']
                    if total_avg > 0:
                        stats[f'ratio_{col}'] = (avg_val / total_avg * 100).round(1)
                    else:
                        stats[f'ratio_{col}'] = 0.0
            
            segment_stats[segment_id] = pd.DataFrame([stats])
        
        return segment_stats
    
    def analyze(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        完整的时间分析流程
        """
        # 添加时段标签
        df_with_segments = self.add_segment_labels(df)
        
        # 分时段统计
        stats = self.segment_statistics(df_with_segments)
        
        return df_with_segments, stats