"""
数据清洗模块
负责无效数据过滤与时序滑动平滑处理
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Dict, Optional


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        self.behavior_columns = [
            'focus_listen', 'study_bow', 'empty_mind', 'sleep_stu', 
            'look_side', 'talk_discuss', 'talk_private',
            'stand_up', 'loose_stu', 'phone_game'
        ]
        # 支持的学生数列名（兼容不同命名）
        self.total_students_columns = ['total_students', 'total_stu']
    
    def _get_total_students_col(self, df: pd.DataFrame) -> str:
        """获取学生总数列名"""
        for col in self.total_students_columns:
            if col in df.columns:
                return col
        return 'total_students'
    
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """加载CSV文件"""
        df = pd.read_csv(csv_path)
        return df
    
    def filter_invalid_frames(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤无效帧
        - 黑屏帧（total_students = 0）
        - 模糊帧（is_valid = False）
        - 异常值（人数为负数）
        """
        total_col = self._get_total_students_col(df)
        
        # 过滤无效帧标记
        if 'is_valid' in df.columns:
            df = df[df['is_valid'] == True]
        
        # 过滤黑屏帧（无学生检测）
        if total_col in df.columns:
            df = df[df[total_col] > 0]
        
        # 过滤负数人数
        for col in self.behavior_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: max(0, x))
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df
    
    def sliding_window_smooth(self, df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
        """
        时序滑动窗口平滑处理
        消除单帧识别偶然误差
        """
        smoothed_df = df.copy()
        
        for col in self.behavior_columns:
            if col in df.columns:
                # 使用滑动窗口平均
                smoothed_df[col] = df[col].rolling(
                    window=window_size, 
                    center=True, 
                    min_periods=1
                ).mean().round().astype(int)
        
        return smoothed_df
    
    def savgol_smooth(self, df: pd.DataFrame, window_length: int = 7, polyorder: int = 2) -> pd.DataFrame:
        """
        使用Savitzky-Golay滤波器进行平滑
        """
        smoothed_df = df.copy()
        
        for col in self.behavior_columns:
            if col in df.columns and len(df) >= window_length:
                try:
                    smoothed_df[col] = savgol_filter(
                        df[col].values, 
                        window_length=window_length, 
                        polyorder=polyorder
                    ).round().astype(int)
                    # 确保数值非负
                    smoothed_df[col] = smoothed_df[col].apply(lambda x: max(0, x))
                except Exception:
                    pass
        
        return smoothed_df
    
    def validate_data_consistency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        验证数据一致性
        - 确保各行为分类总和不超过检测人数
        """
        total_col = self._get_total_students_col(df)
        
        validation_results = {
            'total_frames': len(df),
            'valid_frames': 0,
            'corrected_frames': 0,
            'errors': []
        }
        
        corrected_df = df.copy()
        
        for idx, row in df.iterrows():
            total_students = row.get(total_col, 0)
            behavior_sum = sum(row[col] for col in self.behavior_columns if col in df.columns)
            
            if behavior_sum > total_students:
                # 需要修正
                ratio = total_students / behavior_sum if behavior_sum > 0 else 1
                for col in self.behavior_columns:
                    if col in df.columns:
                        corrected_df.at[idx, col] = int(row[col] * ratio)
                validation_results['corrected_frames'] += 1
                validation_results['errors'].append(f"Frame {idx}: behavior_sum({behavior_sum}) > total_students({total_students})")
            else:
                validation_results['valid_frames'] += 1
        
        return corrected_df, validation_results
    
    def clean(self, csv_path: str, smooth_method: str = 'savgol') -> Tuple[pd.DataFrame, Dict]:
        """
        完整的数据清洗流程
        """
        # 1. 加载数据
        df = self.load_csv(csv_path)
        
        # 2. 过滤无效帧
        df = self.filter_invalid_frames(df)
        
        # 3. 验证数据一致性
        df, validation = self.validate_data_consistency(df)
        
        # 4. 时序平滑
        if smooth_method == 'savgol':
            df = self.savgol_smooth(df)
        else:
            df = self.sliding_window_smooth(df)
        
        # 更新验证信息
        validation['method'] = smooth_method
        validation['final_frames'] = len(df)
        
        return df, validation