"""
趋势分析模块
挖掘课堂注意力衰减幅度、学习状态稳定性、不良行为时序变化等深层趋势特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class TrendAnalyzer:
    """趋势分析器"""
    
    def __init__(self):
        self.behavior_columns = [
            'focus_listen', 'study_bow', 'empty_mind', 'sleep_stu', 
            'look_side', 'talk_discuss', 'talk_private',
            'stand_up', 'loose_stu', 'phone_game'
        ]
        
        # 支持的学生数列名
        self.total_students_columns = ['total_students', 'total_stu']
    
    def _get_total_students_col(self, df):
        """获取学生总数列名"""
        for col in self.total_students_columns:
            if col in df.columns:
                return col
        return 'total_students'
    
    def detect_attention_peaks(self, df: pd.DataFrame, threshold: float = 80.0) -> List[Dict]:
        """
        检测注意力峰值时段
        """
        peaks = []
        
        if 'effective_learning_rate' not in df.columns:
            return peaks
        
        # 使用滑动窗口检测峰值
        window_size = 5
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i:i+window_size]
            avg_rate = window['effective_learning_rate'].mean()
            
            if avg_rate >= threshold:
                peaks.append({
                    'start_frame': i,
                    'end_frame': i + window_size - 1,
                    'start_time': (i * 5) // 60,
                    'end_time': ((i + window_size - 1) * 5) // 60,
                    'avg_effective_rate': round(avg_rate, 1)
                })
        
        return peaks
    
    def detect_attention_dips(self, df: pd.DataFrame, threshold: float = 40.0) -> List[Dict]:
        """
        检测注意力低谷时段
        """
        dips = []
        
        if 'effective_learning_rate' not in df.columns:
            return dips
        
        window_size = 5
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i:i+window_size]
            avg_rate = window['effective_learning_rate'].mean()
            
            if avg_rate <= threshold:
                dips.append({
                    'start_frame': i,
                    'end_frame': i + window_size - 1,
                    'start_time': (i * 5) // 60,
                    'end_time': ((i + window_size - 1) * 5) // 60,
                    'avg_effective_rate': round(avg_rate, 1)
                })
        
        return dips
    
    def analyze_behavior_transitions(self, df: pd.DataFrame) -> Dict:
        """
        分析行为状态转换模式
        """
        transitions = {
            'total_transitions': 0,
            'transition_matrix': {},
            'most_common_transitions': []
        }
        
        # 构建状态转换矩阵
        behavior_cols = [col for col in self.behavior_columns if col in df.columns]
        
        for i in range(len(df) - 1):
            current_state = self._get_dominant_behavior(df.iloc[i], behavior_cols)
            next_state = self._get_dominant_behavior(df.iloc[i+1], behavior_cols)
            
            if current_state and next_state:
                key = f"{current_state}→{next_state}"
                transitions['transition_matrix'][key] = transitions['transition_matrix'].get(key, 0) + 1
                transitions['total_transitions'] += 1
        
        # 获取最常见的转换
        sorted_transitions = sorted(
            transitions['transition_matrix'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        transitions['most_common_transitions'] = [
            {'transition': k, 'count': v} for k, v in sorted_transitions[:5]
        ]
        
        return transitions
    
    def _get_dominant_behavior(self, row: pd.Series, behavior_cols: List[str]) -> str:
        """
        获取某帧的主导行为
        """
        max_val = -1
        max_col = None
        
        for col in behavior_cols:
            if row[col] > max_val:
                max_val = row[col]
                max_col = col
        
        return max_col
    
    def analyze_student_retention(self, df: pd.DataFrame) -> Dict:
        """
        分析学生留存率变化
        """
        total_col = self._get_total_students_col(df)
        
        retention = {
            'initial_count': 0,
            'final_count': 0,
            'retention_rate': 0.0,
            'drop_points': []
        }
        
        if len(df) == 0:
            return retention
        
        retention['initial_count'] = df[total_col].iloc[0]
        retention['final_count'] = df[total_col].iloc[-1]
        
        if retention['initial_count'] > 0:
            retention['retention_rate'] = (retention['final_count'] / retention['initial_count'] * 100).round(1)
        
        # 检测人数骤降点
        for i in range(1, len(df)):
            drop = df[total_col].iloc[i-1] - df[total_col].iloc[i]
            if drop > 2:  # 超过2人离开视为骤降
                retention['drop_points'].append({
                    'frame': i,
                    'time': (i * 5) // 60,
                    'drop_count': drop
                })
        
        return retention
    
    def analyze_rhythm_patterns(self, df: pd.DataFrame) -> Dict:
        """
        分析课堂节奏模式
        """
        patterns = {
            'is_rhythmic': False,
            'periodicity': 0,
            'pattern_description': '',
            'engagement_peaks': []
        }
        
        if 'effective_learning_rate' not in df.columns or len(df) < 10:
            return patterns
        
        # 使用FFT分析周期性
        signal = df['effective_learning_rate'].values
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal))
        
        # 找到主要频率
        positive_freqs = frequencies > 0
        magnitudes = np.abs(fft_result[positive_freqs])
        dominant_freq_idx = np.argmax(magnitudes)
        dominant_freq = frequencies[positive_freqs][dominant_freq_idx]
        
        if dominant_freq > 0:
            period = int(1 / dominant_freq)
            patterns['periodicity'] = period
            
            # 判断是否有明显节奏（周期在5-30帧之间，即25秒到2.5分钟）
            if 5 <= period <= 30:
                patterns['is_rhythmic'] = True
                patterns['pattern_description'] = f"检测到周期性节奏，周期约为{period * 5}秒"
        
        # 检测互动高峰期
        if 'positive_interaction_rate' in df.columns:
            interaction_data = df['positive_interaction_rate']
            threshold = interaction_data.mean() + interaction_data.std()
            
            peaks = df[interaction_data > threshold]
            for _, row in peaks.iterrows():
                patterns['engagement_peaks'].append({
                    'frame': row.name,
                    'time': (row.name * 5) // 60,
                    'interaction_rate': round(row['positive_interaction_rate'], 1)
                })
        
        return patterns
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        完整的趋势分析流程
        """
        return {
            'attention_peaks': self.detect_attention_peaks(df),
            'attention_dips': self.detect_attention_dips(df),
            'behavior_transitions': self.analyze_behavior_transitions(df),
            'student_retention': self.analyze_student_retention(df),
            'rhythm_patterns': self.analyze_rhythm_patterns(df)
        }