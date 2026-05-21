"""
数据分析模块主入口
ClassVision - 课堂专注度智能分析系统

使用示例：
from src.analysis_module import ClassroomAnalyzer

analyzer = ClassroomAnalyzer()
result = analyzer.analyze('cache_csv/your_data.csv')
"""

import os
import pandas as pd
from typing import Dict, Tuple, Optional
from .data_cleaner import DataCleaner
from .temporal_analyzer import TemporalAnalyzer
from .metrics_calculator import MetricsCalculator
from .trend_analyzer import TrendAnalyzer
from .report_generator import ReportGenerator
from .visualization import Visualization


class ClassroomAnalyzer:
    """课堂数据分析器"""
    
    def __init__(self, api_key: str = None):
        self.data_cleaner = DataCleaner()
        self.temporal_analyzer = TemporalAnalyzer()
        self.metrics_calculator = MetricsCalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.report_generator = ReportGenerator(api_key=api_key)
        self.visualization = Visualization()
    
    def analyze(self, csv_path: str, smooth_method: str = 'savgol') -> Dict:
        """
        完整的数据分析流程
        
        参数：
            csv_path: CSV文件路径
            smooth_method: 平滑方法，可选 'savgol' 或 'sliding'
        
        返回：
            包含所有分析结果的字典
        """
        # 1. 数据清洗
        df, validation = self.data_cleaner.clean(csv_path, smooth_method=smooth_method)
        
        # 2. 时间分析（分时段）
        df_with_segments, segment_stats = self.temporal_analyzer.analyze(df)
        
        # 3. 指标计算
        metrics_result = self.metrics_calculator.calculate(df_with_segments)
        df_with_metrics = metrics_result['dataframe']
        
        # 4. 趋势分析
        trend_analysis = self.trend_analyzer.analyze(df_with_metrics)
        
        # 5. 整合结果
        analysis_results = {
            'validation': validation,
            'overall': metrics_result['overall'],
            'trends': metrics_result['trends'],
            'segments': metrics_result['segments'],
            'trend_analysis': trend_analysis
        }
        
        # 6. 生成可视化数据
        visualization_data = self.visualization.generate_all_visualizations(
            df_with_metrics, analysis_results
        )
        
        # 7. 生成报告
        report = self.report_generator.generate_report(analysis_results)
        
        # 保存结果
        self._save_results(analysis_results, visualization_data, report)
        
        return {
            'dataframe': df_with_metrics,
            'analysis': analysis_results,
            'visualization': visualization_data,
            'report': report
        }
    
    def _save_results(self, analysis_results: Dict, visualization_data: Dict, report: Dict):
        """
        保存分析结果到文件
        """
        # 确保目录存在
        os.makedirs('reports', exist_ok=True)
        
        # 保存报告
        report_path = self.report_generator.save_report(report)
        print(f"报告已保存到: {report_path}")
        
        # 保存可视化数据
        viz_path = self.visualization.save_visualization_data(visualization_data)
        print(f"可视化数据已保存到: {viz_path}")
    
    def quick_analyze(self, csv_path: str) -> Dict:
        """
        快速分析 - 返回核心指标
        """
        result = self.analyze(csv_path)
        
        return {
            'avg_effective_learning_rate': result['analysis']['overall']['avg_effective_learning_rate'],
            'avg_distraction_rate': result['analysis']['overall']['avg_distraction_rate'],
            'avg_drowsiness_rate': result['analysis']['overall']['avg_drowsiness_rate'],
            'attention_decay_rate': result['analysis']['trends']['attention_decay_rate'],
            'report_summary': result['report']['report_content'][:500] + '...'
        }


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='课堂数据分析工具')
    parser.add_argument('csv_path', help='CSV文件路径')
    parser.add_argument('--api-key', help='通义千问API Key', default=None)
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ClassroomAnalyzer(api_key=args.api_key)
    
    # 执行分析
    print(f"正在分析: {args.csv_path}")
    result = analyzer.analyze(args.csv_path)
    
    # 输出摘要
    print("\n===== 分析摘要 =====")
    print(f"有效学习率: {result['analysis']['overall']['avg_effective_learning_rate']}%")
    print(f"走神率: {result['analysis']['overall']['avg_distraction_rate']}%")
    print(f"困倦率: {result['analysis']['overall']['avg_drowsiness_rate']}%")
    print(f"注意力衰减率: {result['analysis']['trends']['attention_decay_rate']}%")
    print("\n报告已生成，请查看 reports/ 目录")


if __name__ == '__main__':
    main()