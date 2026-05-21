"""
可视化模块
生成统计图表数据，为可视化学情看板搭建提供数据支撑
"""

import json
from typing import Dict, Any
from datetime import datetime


class Visualization:
    """可视化数据生成器"""
    
    def __init__(self):
        self.color_palette = {
            'focus_listen': '#00d4ff',
            'study_bow': '#10b981',
            'sleep_stu': '#ef4444',
            'look_side': '#f59e0b',
            'stand_up': '#ec4899',
            'loose_stu': '#8b5cf6',
            'raise_hand': '#06b6d4',
            'answer_q': '#22c55e',
            'discussion': '#3b82f6',
            'whisper': '#f97316',
            'opening': '#00d4ff',
            'efficient': '#10b981',
            'fatigue': '#f59e0b',
            'closing': '#ef4444'
        }
        
        self.behavior_labels = {
            'focus_listen': '专注听课',
            'study_bow': '低头学习',
            'sleep_stu': '打瞌睡',
            'look_side': '侧视走神',
            'stand_up': '站立',
            'loose_stu': '松散状态',
            'raise_hand': '举手',
            'answer_q': '回答问题',
            'discussion': '课堂讨论',
            'whisper': '私下闲聊'
        }
    
    def generate_pie_chart_data(self, analysis_results: Dict) -> Dict:
        """
        生成饼图数据（行为分布）
        """
        overall = analysis_results.get('overall', {})
        
        # 计算各类行为的总体比例
        behavior_ratios = {
            'focus_listen': overall.get('ratio_focus_listen', 0) or 0,
            'study_bow': overall.get('ratio_study_bow', 0) or 0,
            'sleep_stu': overall.get('ratio_sleep_stu', 0) or 0,
            'look_side': overall.get('ratio_look_side', 0) or 0,
            'stand_up': overall.get('ratio_stand_up', 0) or 0,
            'loose_stu': overall.get('ratio_loose_stu', 0) or 0,
            'raise_hand': overall.get('ratio_raise_hand', 0) or 0,
            'answer_q': overall.get('ratio_answer_q', 0) or 0,
            'discussion': overall.get('ratio_discussion', 0) or 0,
            'whisper': overall.get('ratio_whisper', 0) or 0
        }
        
        # 过滤掉比例为0的行为
        filtered_data = [
            {
                'name': self.behavior_labels.get(key, key),
                'value': value,
                'color': self.color_palette.get(key, '#6b7280')
            }
            for key, value in behavior_ratios.items()
            if value > 0
        ]
        
        return {
            'type': 'pie',
            'title': '课堂行为分布',
            'data': filtered_data
        }
    
    def generate_bar_chart_data(self, analysis_results: Dict) -> Dict:
        """
        生成柱状图数据（分时段对比）
        """
        segments = analysis_results.get('segments', {})
        
        segment_order = ['opening', 'efficient', 'fatigue', 'closing']
        segment_names = ['开课适应期', '高效学习期', '学习疲劳期', '下课涣散期']
        
        categories = []
        effective_rates = []
        distraction_rates = []
        drowsiness_rates = []
        colors = []
        
        for segment_id in segment_order:
            if segment_id in segments:
                seg = segments[segment_id]
                categories.append(seg.get('name', segment_names[segment_order.index(segment_id)]))
                effective_rates.append(seg.get('avg_effective_learning_rate', 0))
                distraction_rates.append(seg.get('avg_distraction_rate', 0))
                drowsiness_rates.append(seg.get('avg_drowsiness_rate', 0))
                colors.append(self.color_palette.get(segment_id, '#6b7280'))
        
        return {
            'type': 'bar',
            'title': '各时段学习状态对比',
            'categories': categories,
            'series': [
                {
                    'name': '有效学习率',
                    'data': effective_rates,
                    'color': '#10b981'
                },
                {
                    'name': '走神率',
                    'data': distraction_rates,
                    'color': '#f59e0b'
                },
                {
                    'name': '困倦率',
                    'data': drowsiness_rates,
                    'color': '#ef4444'
                }
            ]
        }
    
    def generate_line_chart_data(self, df, analysis_results: Dict) -> Dict:
        """
        生成折线图数据（时序趋势）
        """
        if df is None or len(df) == 0:
            return {
                'type': 'line',
                'title': '学习状态时序变化',
                'data': []
            }
        
        timestamps = [(i * 5) // 60 for i in range(len(df))]  # 转换为分钟
        
        return {
            'type': 'line',
            'title': '学习状态时序变化',
            'x_axis': timestamps,
            'x_label': '时间（分钟）',
            'series': [
                {
                    'name': '有效学习率',
                    'data': df['effective_learning_rate'].tolist(),
                    'color': '#10b981'
                },
                {
                    'name': '走神率',
                    'data': df['distraction_rate'].tolist(),
                    'color': '#f59e0b'
                },
                {
                    'name': '困倦率',
                    'data': df['drowsiness_rate'].tolist(),
                    'color': '#ef4444'
                }
            ]
        }
    
    def generate_summary_cards(self, analysis_results: Dict) -> Dict:
        """
        生成汇总卡片数据
        """
        overall = analysis_results.get('overall', {})
        trends = analysis_results.get('trends', {})
        
        cards = [
            {
                'title': '有效学习率',
                'value': f"{overall.get('avg_effective_learning_rate', 0)}%",
                'trend': 'positive' if overall.get('avg_effective_learning_rate', 0) >= 60 else 'negative',
                'color': '#10b981'
            },
            {
                'title': '走神率',
                'value': f"{overall.get('avg_distraction_rate', 0)}%",
                'trend': 'negative' if overall.get('avg_distraction_rate', 0) >= 20 else 'positive',
                'color': '#f59e0b'
            },
            {
                'title': '困倦率',
                'value': f"{overall.get('avg_drowsiness_rate', 0)}%",
                'trend': 'negative' if overall.get('avg_drowsiness_rate', 0) >= 10 else 'positive',
                'color': '#ef4444'
            },
            {
                'title': '互动率',
                'value': f"{overall.get('avg_positive_interaction_rate', 0)}%",
                'trend': 'positive' if overall.get('avg_positive_interaction_rate', 0) >= 5 else 'neutral',
                'color': '#3b82f6'
            },
            {
                'title': '注意力衰减',
                'value': f"{trends.get('attention_decay幅度', 0)}%",
                'trend': 'negative' if trends.get('attention_decay幅度', 0) >= 15 else 'positive',
                'color': '#8b5cf6'
            },
            {
                'title': '学习稳定性',
                'value': str(trends.get('learning_stability', 0)),
                'trend': 'negative' if trends.get('learning_stability', 0) >= 15 else 'positive',
                'color': '#06b6d4'
            }
        ]
        
        return {
            'type': 'cards',
            'title': '核心指标概览',
            'data': cards
        }
    
    def generate_all_visualizations(self, df, analysis_results: Dict) -> Dict:
        """
        生成所有可视化数据
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'summary_cards': self.generate_summary_cards(analysis_results),
            'pie_chart': self.generate_pie_chart_data(analysis_results),
            'bar_chart': self.generate_bar_chart_data(analysis_results),
            'line_chart': self.generate_line_chart_data(df, analysis_results)
        }
    
    def save_visualization_data(self, visualization_data: Dict, output_path: str = "reports/") -> str:
        """
        保存可视化数据到JSON文件
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        
        filename = f"{output_path}visualization_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(visualization_data, f, ensure_ascii=False, indent=2)
        
        return filename