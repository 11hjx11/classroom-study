from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import yaml
import cv2
from datetime import datetime
import json
import pandas as pd
import numpy as np

app = Flask(__name__,
            template_folder='frontend/templates',
            static_folder='frontend/static')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs/csv'
app.config['REPORTS_FOLDER'] = 'outputs/reports'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})

    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)

    try:
        from src.cv_module import VideoSampler, StudentDetector, CSVSaver
        result = process_video(video_path, file.filename)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def process_video(video_path, filename):
    config = load_config()

    sampler = VideoSampler(config_path='config.yaml')
    detector = StudentDetector(config)
    saver = CSVSaver(output_dir=app.config['OUTPUT_FOLDER'])

    video_info = sampler.get_video_info(video_path)
    frames = sampler.get_sample_frames(video_path)

    results = []
    for i, frame_info in enumerate(frames):
        result = detector.process_frame(
            frame_info['frame'],
            frame_info['frame_num'],
            frame_info['timestamp']
        )
        result['is_valid'] = frame_info.get('is_valid', True)
        result['invalid_reason'] = frame_info.get('invalid_reason', 'valid')
        results.append(result)

    video_name = os.path.splitext(filename)[0]
    output_path = saver.save_frame_data(results, video_name)

    stats = calculate_stats(results)

    return {
        'video_name': video_name,
        'video_info': video_info,
        'total_frames': len(results),
        'valid_frames': sum(1 for r in results if r.get('is_valid', True)),
        'stats': stats
    }

def load_config():
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {
        'sampling': {'frame_interval_sec': 3, 'max_frames': 500},
        'detection': {'model': 'yolov8n.pt', 'conf_threshold': 0.3},
        'head_pose': {'pitch_up': 15, 'pitch_down': -10, 'yaw_side': 20},
        'output': {'cache_csv': 'outputs/csv/'}
    }

def calculate_stats(results):
    if not results:
        return {}

    behavior_cols = ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
                     'look_side', 'talk_discuss', 'talk_private', 'stand_up',
                     'loose_stu', 'phone_game']

    total_students = sum(r.get('total_stu', 0) for r in results)

    stats = {col: sum(r.get(col, 0) for r in results) for col in behavior_cols}

    if total_students > 0:
        for col in behavior_cols:
            stats[f'{col}_ratio'] = round(stats[col] / total_students * 100, 1)

    stats['total_students'] = total_students
    stats['total_frames'] = len(results)
    stats['engagement_rate'] = round(
        (stats['focus_listen'] + stats['study_bow'] + stats.get('stand_up', 0)) / total_students * 100,
        1
    ) if total_students > 0 else 0

    stats['distraction_rate'] = round(
        (stats['empty_mind'] + stats['sleep_stu'] + stats['look_side'] + stats.get('loose_stu', 0)) / total_students * 100,
        1
    ) if total_students > 0 else 0

    stats['avg_effective_learning_rate'] = stats.get('focus_listen_ratio', 0) + stats.get('study_bow_ratio', 0)
    stats['avg_distraction_rate'] = stats.get('empty_mind_ratio', 0) + stats.get('look_side_ratio', 0)
    stats['avg_drowsiness_rate'] = stats.get('sleep_stu_ratio', 0)
    stats['avg_positive_interaction_rate'] = stats.get('talk_discuss_ratio', 0)
    stats['avg_misbehavior_rate'] = stats.get('talk_private_ratio', 0) + stats.get('phone_game_ratio', 0)

    stats['attention_decay_rate'] = calculate_attention_decay(results)

    return stats

def calculate_attention_decay(results):
    if len(results) < 10:
        return 0

    mid = len(results) // 2
    first_half = results[:mid]
    second_half = results[mid:]

    first_rate = sum(r.get('focus_listen', 0) + r.get('study_bow', 0) for r in first_half) / len(first_half)
    second_rate = sum(r.get('focus_listen', 0) + r.get('study_bow', 0) for r in second_half) / len(second_half)

    if first_rate > 0:
        return round((first_rate - second_rate) / first_rate * 100, 1)
    return 0

@app.route('/api/history')
def get_history():
    csv_files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if f.endswith('.csv')]
    history = []
    for file in csv_files:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], file)
        stats = os.stat(filepath)
        history.append({
            'name': file.replace('_raw.csv', ''),
            'filename': file,
            'size': stats.st_size,
            'modified': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        })
    return jsonify(history)

@app.route('/api/analyze/<filename>')
def analyze_file(filename):
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], f'{filename}_raw.csv')
    if not os.path.exists(filepath):
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'})

    df = pd.read_csv(filepath)

    total_col = 'total_students' if 'total_students' in df.columns else 'total_stu'

    behavior_cols = ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
                     'look_side', 'talk_discuss', 'talk_private', 'stand_up',
                     'loose_stu', 'phone_game']

    df['effective_learning_rate'] = ((df.get('focus_listen', 0) + df.get('study_bow', 0)) / df[total_col] * 100).fillna(0).round(1)
    df['distraction_rate'] = ((df.get('empty_mind', 0) + df.get('look_side', 0)) / df[total_col] * 100).fillna(0).round(1)
    df['drowsiness_rate'] = (df.get('sleep_stu', 0) / df[total_col] * 100).fillna(0).round(1)
    df['interaction_rate'] = (df.get('talk_discuss', 0) / df[total_col] * 100).fillna(0).round(1)
    df['misbehavior_rate'] = ((df.get('talk_private', 0) + df.get('phone_game', 0)) / df[total_col] * 100).fillna(0).round(1)

    segments = calculate_segments(df)

    stats = {
        'total_frames': len(df),
        'avg_total_students': round(df[total_col].mean(), 1),
        'avg_effective_learning_rate': round(df['effective_learning_rate'].mean(), 1),
        'avg_distraction_rate': round(df['distraction_rate'].mean(), 1),
        'avg_drowsiness_rate': round(df['drowsiness_rate'].mean(), 1),
        'avg_positive_interaction_rate': round(df['interaction_rate'].mean(), 1),
        'avg_misbehavior_rate': round(df['misbehavior_rate'].mean(), 1),
        'attention_decay_rate': calculate_decay_from_df(df),
        'segments': segments
    }

    for col in behavior_cols:
        if col in df.columns:
            stats[col] = int(df[col].sum())
            stats[f'{col}_ratio'] = round(df[col].sum() / df[total_col].sum() * 100, 1) if df[total_col].sum() > 0 else 0

    data = df.to_dict('records')

    return jsonify({
        'success': True,
        'data': data,
        'stats': stats,
        'segments': segments,
        'columns': df.columns.tolist()
    })

def calculate_segments(df):
    if len(df) == 0:
        return {}

    total_col = 'total_students' if 'total_students' in df.columns else 'total_stu'
    total_frames = len(df)

    segment_ranges = {
        'opening': (0, int(total_frames * 0.2)),
        'efficient': (int(total_frames * 0.2), int(total_frames * 0.6)),
        'fatigue': (int(total_frames * 0.6), int(total_frames * 0.8)),
        'closing': (int(total_frames * 0.8), total_frames)
    }

    segments = {}
    for seg_id, (start, end) in segment_ranges.items():
        segment_data = df.iloc[start:end]
        if len(segment_data) > 0:
            segments[seg_id] = {
                'name': {'opening': '开课适应期', 'efficient': '高效学习期', 'fatigue': '疲劳下滑期', 'closing': '下课涣散期'}[seg_id],
                'frame_count': len(segment_data),
                'avg_total_students': round(segment_data[total_col].mean(), 1),
                'avg_effective_learning_rate': round(segment_data['effective_learning_rate'].mean(), 1) if 'effective_learning_rate' in segment_data else 0,
                'avg_distraction_rate': round(segment_data['distraction_rate'].mean(), 1) if 'distraction_rate' in segment_data else 0,
                'avg_drowsiness_rate': round(segment_data['drowsiness_rate'].mean(), 1) if 'drowsiness_rate' in segment_data else 0,
                'avg_positive_interaction_rate': round(segment_data['interaction_rate'].mean(), 1) if 'interaction_rate' in segment_data else 0,
                'avg_misbehavior_rate': round(segment_data['misbehavior_rate'].mean(), 1) if 'misbehavior_rate' in segment_data else 0
            }

    return segments

def calculate_decay_from_df(df):
    if len(df) < 10:
        return 0

    mid = len(df) // 2
    first_rate = df['effective_learning_rate'].iloc[:mid].mean()
    second_rate = df['effective_learning_rate'].iloc[mid:].mean()

    if first_rate > 0:
        return round((first_rate - second_rate) / first_rate * 100, 1)
    return 0

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    data = request.get_json()
    stats = data.get('stats', {})

    try:
        report = generate_ai_report(stats)

        report_filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        return jsonify({
            'success': True,
            'report': report,
            'filename': report_filename
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def generate_ai_report(stats):
    api_key = "sk-02bf14d117fb415bbc28e7ce41a4c9db"
    model = "qwen-plus-2025-07-28"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    effective_rate = stats.get('avg_effective_learning_rate', 0)
    distraction_rate = stats.get('avg_distraction_rate', 0)
    drowsiness_rate = stats.get('avg_drowsiness_rate', 0)
    interaction_rate = stats.get('avg_positive_interaction_rate', 0)
    misbehavior_rate = stats.get('avg_misbehavior_rate', 0)
    decay_rate = stats.get('attention_decay_rate', 0)
    segments = stats.get('segments', {})

    prompt = f"""【角色设定】
你是资深教学督导、课堂学情数据分析专家，熟悉中小学课堂评价标准，语言严谨客观、通俗易懂、贴合教研文风。

【强制判别规则（必须严格遵守）】
1. 低头行为二分判定：低头手部伏案、书写看书判定为正常学习行为；无动作慵懒低头判定为放空走神，严禁混淆。
2. 交谈行为二分判定：全班大面积多人同步交流判定为教师组织合规自由讨论；零散个别学生两两交谈判定为私下闲聊违纪行为。
3. 严格按照课堂四段划分：开课适应期、高效学习期、疲劳下滑期、下课涣散期逐段分析变化规律。
4. 所有分析必须依据给定数据，禁止编造数据、禁止主观臆断。
5. 教学建议要贴合真实课堂、简短落地、不空洞、不官方套话。

【固定输出报告结构，不可更改】
一、课堂基础数据概况
二、班级整体学情综合评价
三、四大授课时段学情分析
四、学生课堂行为结构分析
五、课堂现存优势与突出问题
六、教学优化与班级管理建议
七、综合评分与课堂等级评定

【输入数据】
- 有效学习率（平均）：{effective_rate}%
- 走神率（平均）：{distraction_rate}%
- 困倦率（平均）：{drowsiness_rate}%
- 课堂互动率（平均）：{interaction_rate}%
- 违纪率（平均）：{misbehavior_rate}%
- 注意力衰减率：{decay_rate}%

【时段数据】
"""

    for seg_id, seg_name in [('opening', '开课适应期'), ('efficient', '高效学习期'),
                             ('fatigue', '疲劳下滑期'), ('closing', '下课涣散期')]:
        if seg_id in segments:
            seg = segments[seg_id]
            prompt += f"{seg_name}：有效学习率{seg.get('avg_effective_learning_rate', 0)}%，走神率{seg.get('avg_distraction_rate', 0)}%，困倦率{seg.get('avg_drowsiness_rate', 0)}%\n"

    prompt += """
【输出要求】
全文正式、条理清晰、段落分明，不用markdown，不用加粗，直接生成可存档的正式学情分析报告。
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 4000
    }

    try:
        import requests
        response = requests.post(base_url, headers=headers, json=payload, timeout=60)
        result = response.json()

        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        elif 'output' in result:
            if isinstance(result['output'], dict) and 'text' in result['output']:
                return result['output']['text']
            elif isinstance(result['output'], str):
                return result['output']

        return f"[报告生成异常，使用模拟数据]\n\n有效学习率: {effective_rate}%\n走神率: {distraction_rate}%\n困倦率: {drowsiness_rate}%\n互动率: {interaction_rate}%\n违纪率: {misbehavior_rate}%\n注意力衰减率: {decay_rate}%"
    except Exception as e:
        return f"[AI报告生成失败: {str(e)}\n\n模拟报告]\n\n有效学习率: {effective_rate}%\n走神率: {distraction_rate}%\n困倦率: {drowsiness_rate}%\n互动率: {interaction_rate}%\n违纪率: {misbehavior_rate}%\n注意力衰减率: {decay_rate}%"

@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(app.config['REPORTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)