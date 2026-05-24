"""
指标计算模块
批量计算多维度学情量化指标
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        # 行为分类定义（与CSV输出一致）
        self.behavior_definitions = {
            # 有效学习行为
            'effective_learning': ['focus_listen', 'study_bow'],
            # 走神行为
            'distraction': ['look_side', 'empty_mind'],
            # 困倦行为
            'drowsiness': ['sleep_stu'],
            # 合规互动行为
            'positive_interaction': ['talk_discuss'],
            # 违纪行为
            'misbehavior': ['stand_up', 'talk_private', 'phone_game'],
            # 所有行为
            'all': ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu', 
                    'look_side', 'talk_discuss', 'talk_private',
                    'stand_up', 'loose_stu', 'phone_game']
        }
        
        # 支持的学生数列名
        self.total_students_columns = ['total_students', 'total_stu']
    
    def _get_total_students_col(self, df):
        """获取学生总数列名"""
        for col in self.total_students_columns:
            if col in df.columns:
                return col
        return 'total_students'
    
    def calculate_frame_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算每帧的基础指标
        """
        df = df.copy()
        total_col = self._get_total_students_col(df)
        
        # 计算各类行为总和（只计算存在的列）
        for category, behaviors in self.behavior_definitions.items():
            existing_behaviors = [b for b in behaviors if b in df.columns]
            if existing_behaviors:
                df[f'{category}_sum'] = df[existing_behaviors].sum(axis=1)
            else:
                df[f'{category}_sum'] = 0
        
        # 计算各率值（基于该帧总人数）
        total_students = df[total_col]
        df['effective_learning_rate'] = ((df['effective_learning_sum'] / total_students) * 100).round(1)
        df['distraction_rate'] = ((df['distraction_sum'] / total_students) * 100).round(1)
        df['drowsiness_rate'] = ((df['drowsiness_sum'] / total_students) * 100).round(1)
        df['positive_interaction_rate'] = ((df['positive_interaction_sum'] / total_students) * 100).round(1)
        df['misbehavior_rate'] = ((df['misbehavior_sum'] / total_students) * 100).round(1)
        
        # 处理除零情况
        df = df.fillna(0)
        
        return df
    
    def calculate_overall_metrics(self, df: pd.DataFrame) -> Dict:
        """
        计算整节课的总体指标
        """
        metrics = {}
        total_col = self._get_total_students_col(df)
        
        # 基础统计
        metrics['total_frames'] = len(df)
        metrics['avg_total_students'] = df[total_col].mean().round(1)
        metrics['max_total_students'] = df[total_col].max()
        metrics['min_total_students'] = df[total_col].min()
        
        # 有效学习率（整节课平均）
        metrics['avg_effective_learning_rate'] = df['effective_learning_rate'].mean().round(1)
        metrics['max_effective_learning_rate'] = df['effective_learning_rate'].max()
        metrics['min_effective_learning_rate'] = df['effective_learning_rate'].min()
        
        # 走神率
        metrics['avg_distraction_rate'] = df['distraction_rate'].mean().round(1)
        metrics['max_distraction_rate'] = df['distraction_rate'].max()
        
        # 困倦率
        metrics['avg_drowsiness_rate'] = df['drowsiness_rate'].mean().round(1)
        metrics['max_drowsiness_rate'] = df['drowsiness_rate'].max()
        
        # 互动率
        metrics['avg_positive_interaction_rate'] = df['positive_interaction_rate'].mean().round(1)
        
        # 违纪率
        metrics['avg_misbehavior_rate'] = df['misbehavior_rate'].mean().round(1)
        
        return metrics
    
    def calculate_temporal_trends(self, df: pd.DataFrame) -> Dict:
        """
        计算时序趋势指标
        """
        trends = {}
        
        # 注意力衰减幅度（高效学习期 vs 学习疲劳期）
        if 'segment' in df.columns:
            efficient_data = df[df['segment'] == 'efficient']
            fatigue_data = df[df['segment'] == 'fatigue']
            
            if len(efficient_data) > 0 and len(fatigue_data) > 0:
                efficient_learning_rate = efficient_data['effective_learning_rate'].mean()
                fatigue_learning_rate = fatigue_data['effective_learning_rate'].mean()
                trends['attention_decay幅度'] = (efficient_learning_rate - fatigue_learning_rate).round(1)
                trends['attention_decay_rate'] = ((efficient_learning_rate - fatigue_learning_rate) / efficient_learning_rate * 100).round(1)
            else:
                trends['attention_decay幅度'] = 0.0
                trends['attention_decay_rate'] = 0.0
        else:
            # 如果没有时段标签，使用前后半段对比
            mid_point = len(df) // 2
            first_half = df.iloc[:mid_point]
            second_half = df.iloc[mid_point:]
            
            if len(first_half) > 0 and len(second_half) > 0:
                first_rate = first_half['effective_learning_rate'].mean()
                second_rate = second_half['effective_learning_rate'].mean()
                trends['attention_decay幅度'] = (first_rate - second_rate).round(1)
                trends['attention_decay_rate'] = ((first_rate - second_rate) / first_rate * 100).round(1)
            else:
                trends['attention_decay幅度'] = 0.0
                trends['attention_decay_rate'] = 0.0
        
        # 学习状态稳定性（有效学习率的标准差）
        trends['learning_stability'] = df['effective_learning_rate'].std().round(1)
        
        # 不良行为时序变化
        if 'segment' in df.columns:
            opening_misbehavior = df[df['segment'] == 'opening']['misbehavior_rate'].mean()
            closing_misbehavior = df[df['segment'] == 'closing']['misbehavior_rate'].mean()
            trends['misbehavior_increase'] = (closing_misbehavior - opening_misbehavior).round(1)
            trends['misbehavior_increase_rate'] = ((closing_misbehavior - opening_misbehavior) / max(opening_misbehavior, 1) * 100).round(1)
        else:
            trends['misbehavior_increase'] = 0.0
            trends['misbehavior_increase_rate'] = 0.0
        
        return trends
    
    def calculate_segment_metrics(self, df: pd.DataFrame) -> Dict:
        """
        计算各时段的指标
        """
        total_col = self._get_total_students_col(df)
        
        segment_metrics = {}
        segments = ['opening', 'efficient', 'fatigue', 'closing']
        segment_names = ['上课初期', '上课关键期', '疲劳期', '临近下课期']
        
        for segment_id, segment_name in zip(segments, segment_names):
            segment_data = df[df['segment'] == segment_id]
            
            if len(segment_data) == 0:
                continue
            
            metrics = {
                'name': segment_name,
                'frame_count': len(segment_data),
                'avg_total_students': segment_data[total_col].mean().round(1),
                'avg_effective_learning_rate': segment_data['effective_learning_rate'].mean().round(1),
                'avg_distraction_rate': segment_data['distraction_rate'].mean().round(1),
                'avg_drowsiness_rate': segment_data['drowsiness_rate'].mean().round(1),
                'avg_positive_interaction_rate': segment_data['positive_interaction_rate'].mean().round(1),
                'avg_misbehavior_rate': segment_data['misbehavior_rate'].mean().round(1)
            }
            
            segment_metrics[segment_id] = metrics
        
        return segment_metrics
    
    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        完整的指标计算流程
        """
        # 计算每帧指标
        df_with_metrics = self.calculate_frame_metrics(df)
        
        # 计算总体指标
        overall = self.calculate_overall_metrics(df_with_metrics)
        
        # 计算时序趋势
        trends = self.calculate_temporal_trends(df_with_metrics)
        
        # 计算时段指标
        segment = self.calculate_segment_metrics(df_with_metrics)
        
        return {
            'dataframe': df_with_metrics,
            'overall': overall,
            'trends': trends,
            'segments': segment
        }