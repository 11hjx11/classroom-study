"""
课堂专注度分析系统 - 可视化图表与自动分析报告生成
功能：读取CSV数据，生成图表和文本报告
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BEHAVIOR_NAMES = {
    'focus_listen': '专注听课',
    'study_bow': '低头研学',
    'empty_mind': '低头放空',
    'sleep_stu': '趴桌犯困',
    'look_side': '张望',
    'talk_discuss': '集体讨论',
    'talk_private': '私下闲聊',
    'stand_up': '站立互动',
    'loose_stu': '散漫游离',
    'phone_game': '玩手机'
}

BEHAVIOR_COLORS = {
    'focus_listen': '#2ECC71',
    'study_bow': '#3498DB',
    'empty_mind': '#E67E22',
    'sleep_stu': '#E74C3C',
    'look_side': '#9B59B6',
    'talk_discuss': '#1ABC9C',
    'talk_private': '#F39C12',
    'stand_up': '#2980B9',
    'loose_stu': '#95A5A6',
    'phone_game': '#C0392B'
}


def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"文件不存在: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    print(f"✅ 加载数据: {len(df)} 帧, {df['total_stu'].sum()} 人次")
    return df


def calculate_metrics(df):
    if df.empty:
        return {}

    total_frames = len(df)
    total_person_times = df['total_stu'].sum()

    behavior_cols = [col for col in BEHAVIOR_NAMES.keys() if col in df.columns]

    behavior_counts = {}
    for col in behavior_cols:
        behavior_counts[col] = df[col].sum()

    behavior_ratios = {}
    if total_person_times > 0:
        for col, count in behavior_counts.items():
            behavior_ratios[col] = round(count / total_person_times * 100, 1)

    engagement_rate = (
        behavior_ratios.get('focus_listen', 0) +
        behavior_ratios.get('study_bow', 0) +
        behavior_ratios.get('stand_up', 0)
    )

    distraction_rate = (
        behavior_ratios.get('empty_mind', 0) +
        behavior_ratios.get('sleep_stu', 0) +
        behavior_ratios.get('look_side', 0) +
        behavior_ratios.get('loose_stu', 0) +
        behavior_ratios.get('phone_game', 0)
    )

    violation_rate = (
        behavior_ratios.get('talk_private', 0) +
        behavior_ratios.get('phone_game', 0)
    )

    return {
        'total_frames': total_frames,
        'total_person_times': total_person_times,
        'behavior_counts': behavior_counts,
        'behavior_ratios': behavior_ratios,
        'engagement_rate': round(engagement_rate, 1),
        'distraction_rate': round(distraction_rate, 1),
        'violation_rate': round(violation_rate, 1)
    }


def calculate_segment_stats(df, duration, segments):
    segment_results = {}

    for seg in segments:
        start_ratio, end_ratio = seg['ratio']
        start_time = duration * start_ratio
        end_time = duration * end_ratio

        seg_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]

        if seg_df.empty:
            segment_results[seg['name']] = None
            continue

        total_person_times = seg_df['total_stu'].sum()

        behavior_cols = [col for col in BEHAVIOR_NAMES.keys() if col in seg_df.columns]

        behavior_ratios = {}
        if total_person_times > 0:
            for col in behavior_cols:
                count = seg_df[col].sum()
                behavior_ratios[col] = round(count / total_person_times * 100, 1)

        engagement_rate = (
            behavior_ratios.get('focus_listen', 0) +
            behavior_ratios.get('study_bow', 0) +
            behavior_ratios.get('stand_up', 0)
        )

        segment_results[seg['name']] = {
            'total_frames': len(seg_df),
            'total_person_times': total_person_times,
            'behavior_ratios': behavior_ratios,
            'engagement_rate': round(engagement_rate, 1)
        }

    return segment_results


def plot_behavior_pie(behavior_ratios, output_path):
    if not behavior_ratios:
        return

    data = {k: v for k, v in behavior_ratios.items() if v > 0}
    if not data:
        return

    labels = [BEHAVIOR_NAMES.get(k, k) for k in data.keys()]
    sizes = list(data.values())
    colors = [BEHAVIOR_COLORS.get(k, '#999999') for k in data.keys()]

    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 11}
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.title('课堂行为分布图', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 饼图: {output_path}")


def plot_engagement_trend(df, output_path):
    if df.empty:
        return

    window = 30
    df['time_window'] = (df['timestamp'] // window).astype(int)

    trend_data = []
    for window_id in df['time_window'].unique():
        window_df = df[df['time_window'] == window_id]
        total = window_df['total_stu'].sum()
        if total > 0:
            focus = window_df['focus_listen'].sum()
            study = window_df['study_bow'].sum()
            stand = window_df['stand_up'].sum()
            engagement = (focus + study + stand) / total * 100
        else:
            engagement = 0
        trend_data.append({
            'window': window_id,
            'time': window_id * window,
            'engagement': engagement
        })

    trend_df = pd.DataFrame(trend_data)

    plt.figure(figsize=(12, 5))
    plt.plot(trend_df['time'], trend_df['engagement'], 
             marker='o', linewidth=2, color='#2E86AB', markersize=4)
    plt.fill_between(trend_df['time'], trend_df['engagement'], 0, alpha=0.3, color='#2E86AB')
    plt.xlabel('时间 (秒)', fontsize=12)
    plt.ylabel('课堂参与度 (%)', fontsize=12)
    plt.title('课堂参与度变化趋势', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 趋势图: {output_path}")


def plot_segment_comparison(segment_stats, output_path):
    if not segment_stats:
        return

    segments = list(segment_stats.keys())
    engagement_rates = []
    for seg in segments:
        stats = segment_stats[seg]
        if stats:
            engagement_rates.append(stats['engagement_rate'])
        else:
            engagement_rates.append(0)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(segments, engagement_rates, color='#3498DB', edgecolor='white', linewidth=2)

    for bar, rate in zip(bars, engagement_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xlabel('课堂时段', fontsize=12)
    plt.ylabel('参与度 (%)', fontsize=12)
    plt.title('四时段课堂参与度对比', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 时段对比图: {output_path}")


def generate_text_report(metrics, segment_stats, duration, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("课堂专注度分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("【视频信息】\n")
        f.write(f"  视频时长: {duration:.1f} 秒 ({duration/60:.1f} 分钟)\n")
        f.write(f"  分析帧数: {metrics.get('total_frames', 0)}\n")
        f.write(f"  总人次数: {metrics.get('total_person_times', 0)}\n\n")

        f.write("【整体指标】\n")
        f.write(f"  课堂参与度: {metrics.get('engagement_rate', 0)}%\n")
        f.write(f"  走神率: {metrics.get('distraction_rate', 0)}%\n")
        f.write(f"  违纪率: {metrics.get('violation_rate', 0)}%\n\n")

        f.write("【行为分布】\n")
        for behavior, ratio in metrics.get('behavior_ratios', {}).items():
            if ratio > 0:
                f.write(f"  {BEHAVIOR_NAMES.get(behavior, behavior)}: {ratio}%\n")
        f.write("\n")

        f.write("【四时段分析】\n")
        for seg_name, stats in segment_stats.items():
            if stats:
                f.write(f"\n  📊 {seg_name}:\n")
                f.write(f"     参与度: {stats['engagement_rate']}%\n")
                f.write(f"     主要行为:\n")
                for behavior, ratio in stats['behavior_ratios'].items():
                    if ratio > 5:
                        f.write(f"       - {BEHAVIOR_NAMES.get(behavior, behavior)}: {ratio}%\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("【分析结论】\n")

        engagement = metrics.get('engagement_rate', 0)
        if engagement >= 80:
            f.write("  课堂氛围良好，学生整体参与度高。\n")
        elif engagement >= 60:
            f.write("  课堂氛围一般，部分学生存在走神现象。\n")
        else:
            f.write("  课堂参与度偏低，需要关注学生的听课状态。\n")

        distraction = metrics.get('distraction_rate', 0)
        if distraction > 30:
            f.write("  走神率较高，建议增加课堂互动环节。\n")

        violation = metrics.get('violation_rate', 0)
        if violation > 10:
            f.write("  违纪行为较多，建议加强课堂纪律管理。\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("报告结束\n")

    print(f"✅ 文本报告: {output_path}")


def main():
    print("=" * 60)
    print("课堂专注度分析系统 - 可视化图表与报告生成")
    print("=" * 60)

    csv_dir = 'cache_csv'
    if not os.path.exists(csv_dir):
        print(f"❌ 找不到 {csv_dir} 文件夹")
        return

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"❌ {csv_dir} 文件夹中没有CSV文件")
        print("请先运行 python main.py 生成数据")
        return

    csv_path = os.path.join(csv_dir, csv_files[0])
    print(f"📁 找到数据: {csv_files[0]}")

    df = load_data(csv_path)
    if df is None:
        return

    if 'timestamp' in df.columns:
        duration = df['timestamp'].max()
    else:
        duration = 128.2

    segments = [
        {"name": "初期", "ratio": [0, 0.25]},
        {"name": "黄金时期", "ratio": [0.25, 0.5]},
        {"name": "疲惫期", "ratio": [0.5, 0.75]},
        {"name": "结尾期", "ratio": [0.75, 1.0]}
    ]

    print("\n正在计算指标...")
    metrics = calculate_metrics(df)
    segment_stats = calculate_segment_stats(df, duration, segments)

    output_dir = 'reports'
    os.makedirs(output_dir, exist_ok=True)

    print("\n正在生成图表...")
    plot_behavior_pie(metrics.get('behavior_ratios', {}), 
                      f"{output_dir}/behavior_pie.png")
    plot_engagement_trend(df, 
                          f"{output_dir}/engagement_trend.png")
    plot_segment_comparison(segment_stats, 
                            f"{output_dir}/segment_comparison.png")

    print("\n正在生成报告...")
    generate_text_report(metrics, segment_stats, duration, 
                        f"{output_dir}/analysis_report.txt")

    print("\n" + "=" * 60)
    print("分析结果汇总")
    print("=" * 60)
    print(f"  课堂参与度: {metrics.get('engagement_rate', 0)}%")
    print(f"  走神率: {metrics.get('distraction_rate', 0)}%")
    print(f"  违纪率: {metrics.get('violation_rate', 0)}%")
    print("\n  行为分布:")
    for behavior, ratio in metrics.get('behavior_ratios', {}).items():
        if ratio > 0:
            print(f"    {BEHAVIOR_NAMES.get(behavior, behavior)}: {ratio}%")
    print("\n" + "=" * 60)
    print(f"📁 报告输出目录: {output_dir}/")
    print("   - behavior_pie.png (行为分布饼图)")
    print("   - engagement_trend.png (参与度趋势图)")
    print("   - segment_comparison.png (时段对比图)")
    print("   - analysis_report.txt (文本分析报告)")
    print("=" * 60)


if __name__ == "__main__":
    main()