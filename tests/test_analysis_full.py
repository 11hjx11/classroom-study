"""
数据分析测试脚本
验证仿真数据的分析流程
"""

from src.analysis_module import ClassroomAnalyzer

def main():
    # 创建分析器
    analyzer = ClassroomAnalyzer(api_key='sk-02bf14d117fb415bbc28e7ce41a4c9db')

    # 加载CSV数据并分析
    print("正在分析仿真数据...")
    result = analyzer.analyze('cache_csv/demo_raw.csv')

    # 输出分析结果摘要
    print('=' * 60)
    print('📊 数据分析结果摘要')
    print('=' * 60)
    print(f"总帧数: {result['analysis']['overall']['total_frames']}")
    # print(f"有效帧数: {result['analysis']['overall']['valid_frames']}")  # 该字段不存在
    print(f"平均检测人数: {result['analysis']['overall']['avg_total_students']:.1f}")
    print()
    print("🎯 核心指标:")
    print(f"  有效学习率: {result['analysis']['overall']['avg_effective_learning_rate']:.1f}%")
    print(f"  走神率: {result['analysis']['overall']['avg_distraction_rate']:.1f}%")
    print(f"  困倦率: {result['analysis']['overall']['avg_drowsiness_rate']:.1f}%")
    print(f"  互动率: {result['analysis']['overall']['avg_positive_interaction_rate']:.1f}%")
    print(f"  违纪率: {result['analysis']['overall']['avg_misbehavior_rate']:.1f}%")
    print()
    print("📈 趋势特征:")
    print(f"  注意力衰减幅度: {result['analysis']['trends']['attention_decay幅度']:.1f}个百分点")
    print(f"  注意力衰减率: {result['analysis']['trends']['attention_decay_rate']:.1f}%")
    print(f"  学习状态稳定性: {result['analysis']['trends']['learning_stability']:.2f}")
    print()
    print("⏰ 时段分析:")
    segments = result['analysis']['segments']
    for seg_id, metrics in segments.items():
        print(f"  {metrics['name']}: 有效学习率 {metrics['avg_effective_learning_rate']:.1f}%")
    
    # 检查是否有报告
    if 'report' in result:
        print()
        print("📝 报告预览:")
        report_preview = result['report']['report_content'][:800]
        print(report_preview)
        if len(result['report']['report_content']) > 800:
            print("...（报告内容继续）")

if __name__ == '__main__':
    main()