"""
数据分析模块测试脚本
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis_module import ClassroomAnalyzer

def test_analysis():
    """测试数据分析流程"""
    # 示例CSV文件路径（使用cache_csv目录中的文件）
    csv_path = 'cache_csv/demo_output.csv'
    
    if not os.path.exists(csv_path):
        print(f"错误：未找到CSV文件 {csv_path}")
        print("请先运行主程序生成CSV数据")
        return
    
    # 创建分析器（使用用户提供的API Key）
    api_key = "sk-02bf14d117fb415bbc28e7ce41a4c9db"
    analyzer = ClassroomAnalyzer(api_key=api_key)
    
    # 执行分析
    print(f"正在分析文件: {csv_path}")
    print("="*50)
    
    try:
        result = analyzer.analyze(csv_path)
        
        # 输出核心指标
        print("\n【核心指标】")
        print(f"有效学习率: {result['analysis']['overall']['avg_effective_learning_rate']}%")
        print(f"走神率: {result['analysis']['overall']['avg_distraction_rate']}%")
        print(f"困倦率: {result['analysis']['overall']['avg_drowsiness_rate']}%")
        print(f"互动率: {result['analysis']['overall']['avg_positive_interaction_rate']}%")
        print(f"违纪率: {result['analysis']['overall']['avg_misbehavior_rate']}%")
        
        print("\n【趋势特征】")
        print(f"注意力衰减幅度: {result['analysis']['trends']['attention_decay幅度']}个百分点")
        print(f"注意力衰减率: {result['analysis']['trends']['attention_decay_rate']}%")
        print(f"学习状态稳定性: {result['analysis']['trends']['learning_stability']}")
        
        print("\n【时段分析】")
        segments = result['analysis']['segments']
        for seg_id, seg_data in segments.items():
            print(f"- {seg_data['name']}: 有效学习率 {seg_data['avg_effective_learning_rate']}%")
        
        print("\n【报告预览】")
        print(result['report']['report_content'][:800] + "..." if len(result['report']['report_content']) > 800 else result['report']['report_content'])
        
        print("\n" + "="*50)
        print("分析完成！结果已保存到 reports/ 目录")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_analysis()