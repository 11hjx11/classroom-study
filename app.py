from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import yaml
import cv2
from datetime import datetime
import json

app = Flask(__name__,
            template_folder='frontend/templates',
            static_folder='frontend/static')

# 配置
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'cache_csv'
app.config['REPORTS_FOLDER'] = 'reports'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cv_module import VideoSampler, StudentDetector, CSVSaver

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # 保存上传的视频
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)
    
    # 处理视频
    try:
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
    
    # 生成统计数据
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
        'output': {'cache_csv': 'cache_csv/'}
    }

def calculate_stats(results):
    if not results:
        return {}
    
    total_students = sum(r['total_stu'] for r in results)
    behavior_cols = ['focus_listen', 'study_bow', 'empty_mind', 'sleep_stu',
                     'look_side', 'talk_discuss', 'talk_private', 'stand_up',
                     'loose_stu', 'phone_game']
    
    stats = {col: sum(r.get(col, 0) for r in results) for col in behavior_cols}
    
    # 计算比率
    if total_students > 0:
        for col in behavior_cols:
            stats[f'{col}_ratio'] = round(stats[col] / total_students * 100, 1)
    
    stats['total_students'] = total_students
    stats['engagement_rate'] = round(
        (stats['focus_listen'] + stats['study_bow'] + stats['stand_up']) / total_students * 100,
        1
    ) if total_students > 0 else 0
    
    stats['distraction_rate'] = round(
        (stats['empty_mind'] + stats['sleep_stu'] + stats['look_side'] + stats['loose_stu']) / total_students * 100,
        1
    ) if total_students > 0 else 0
    
    return stats

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
    import pandas as pd
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], f'{filename}_raw.csv')
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'})
    
    df = pd.read_csv(filepath)
    data = df.to_dict('records')
    
    # 计算统计
    stats = calculate_stats(data)
    
    return jsonify({
        'success': True,
        'data': data,
        'stats': stats,
        'columns': df.columns.tolist()
    })

@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(app.config['REPORTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)