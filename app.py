from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import yaml
import cv2
import json
import numpy as np

def convert_to_json_serializable(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return obj
from datetime import datetime
import pandas as pd

app = Flask(__name__,
            template_folder='frontend/templates',
            static_folder='frontend/static')

# 设置请求大小限制（500MB）
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# 添加 CORS 支持
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# 处理 OPTIONS 请求（CORS预检）
@app.route('/api/auto_analyze', methods=['OPTIONS'])
def handle_options():
    return jsonify({'success': True}), 200

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'cache_csv'
app.config['REPORTS_FOLDER'] = 'outputs/reports'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)

    return jsonify({
        'success': True,
        'message': 'Upload successful',
        'filename': file.filename
    })

@app.route('/api/auto_analyze', methods=['POST'])
def auto_analyze():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})

        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        video_name = os.path.splitext(file.filename)[0]
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)

        # Step 1: 视频采样
        from src.cv_module.video_sampler import VideoSampler
        from src.cv_module.student_detector import StudentDetector
        from src.cv_module.csv_saver import CSVSaver
        
        sampler = VideoSampler(config_path='config.yaml')
        detector = StudentDetector(load_config())
        saver = CSVSaver(output_dir=app.config['OUTPUT_FOLDER'])

        video_info = sampler.get_video_info(video_path)
        frames = sampler.get_sample_frames(video_path)

        # Step 2: 帧检测
        results = []
        for frame_info in frames:
            result = detector.process_frame(
                frame_info['frame'],
                frame_info['frame_num'],
                frame_info['timestamp']
            )
            results.append(result)

        output_path = saver.save_frame_data(results, video_name)

        # Step 3: 数据分析
        df = pd.read_csv(output_path)
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

        # 计算总学生数（用于计算比率）
        total_students_sum = df[total_col].sum()
        
        # 计算参与率（专注听讲 + 低头学习 + 站立）
        focus_listen_sum = df.get('focus_listen', 0).sum() if 'focus_listen' in df.columns else 0
        study_bow_sum = df.get('study_bow', 0).sum() if 'study_bow' in df.columns else 0
        stand_up_sum = df.get('stand_up', 0).sum() if 'stand_up' in df.columns else 0
        engagement_rate = round((focus_listen_sum + study_bow_sum + stand_up_sum) / total_students_sum * 100, 1) if total_students_sum > 0 else 0
        
        # 计算分心率（放空 + 瞌睡 + 左顾右盼 + 松散坐姿）
        empty_mind_sum = df.get('empty_mind', 0).sum() if 'empty_mind' in df.columns else 0
        sleep_stu_sum = df.get('sleep_stu', 0).sum() if 'sleep_stu' in df.columns else 0
        look_side_sum = df.get('look_side', 0).sum() if 'look_side' in df.columns else 0
        loose_stu_sum = df.get('loose_stu', 0).sum() if 'loose_stu' in df.columns else 0
        distraction_rate = round((empty_mind_sum + sleep_stu_sum + look_side_sum + loose_stu_sum) / total_students_sum * 100, 1) if total_students_sum > 0 else 0
        
        stats = {
            'total_frames': len(df),
            'valid_frames': len(df),
            'total_students': total_students_sum,
            'avg_per_frame': round(total_students_sum / len(df), 1) if len(df) > 0 else 0,
            'avg_total_students': round(df[total_col].mean(), 1),
            'avg_effective_learning_rate': round(df['effective_learning_rate'].mean(), 1),
            'avg_distraction_rate': round(df['distraction_rate'].mean(), 1),
            'avg_drowsiness_rate': round(df['drowsiness_rate'].mean(), 1),
            'avg_positive_interaction_rate': round(df['interaction_rate'].mean(), 1),
            'avg_misbehavior_rate': round(df['misbehavior_rate'].mean(), 1),
            'attention_decay_rate': calculate_decay_from_df(df),
            'segments': segments,
            'engagement_rate': engagement_rate,
            'distraction_rate': distraction_rate,
            'sleep_count': int(sleep_stu_sum)
        }

        for col in behavior_cols:
            if col in df.columns:
                stats[col] = int(df[col].sum())
                stats[f'{col}_ratio'] = round(df[col].sum() / df[total_col].sum() * 100, 1) if df[total_col].sum() > 0 else 0

        # 添加 behaviors 结构以匹配前端期望
        stats['behaviors'] = {}
        for col in behavior_cols:
            if col in df.columns:
                count = int(df[col].sum())
                ratio = round(df[col].sum() / df[total_col].sum() * 100, 1) if df[total_col].sum() > 0 else 0
                stats['behaviors'][col] = {'count': count, 'ratio': ratio}
            else:
                stats['behaviors'][col] = {'count': 0, 'ratio': 0}

        # Step 4: AI报告生成
        from src.analysis_module.report_generator import ReportGenerator
        # 使用用户提供的API密钥
        api_key = "sk-02bf14d117fb415bbc28e7ce41a4c9db"
        generator = ReportGenerator(api_key=api_key)
        
        # 构建完整的分析结果数据
        analysis_results = {
            'overall': {
                'total_frames': stats.get('total_frames', 0),
                'avg_total_students': stats.get('avg_total_students', 0),
                'avg_effective_learning_rate': stats.get('avg_effective_learning_rate', 0),
                'avg_distraction_rate': stats.get('avg_distraction_rate', 0),
                'avg_drowsiness_rate': stats.get('avg_drowsiness_rate', 0),
                'avg_positive_interaction_rate': stats.get('avg_positive_interaction_rate', 0),
                'avg_misbehavior_rate': stats.get('avg_misbehavior_rate', 0),
                'max_effective_learning_rate': stats.get('avg_effective_learning_rate', 0),
                'min_effective_learning_rate': stats.get('avg_effective_learning_rate', 0)
            },
            'trends': {
                'attention_decay幅度': stats.get('attention_decay_rate', 0),
                'attention_decay_rate': stats.get('attention_decay_rate', 0),
                'learning_stability': 0,
                'misbehavior_increase': 0
            },
            'segments': {},
            'trend_analysis': {}
        }
        
        # 添加时段数据
        if stats.get('segments'):
            for seg in stats['segments']:
                seg_id = seg.get('name', '').replace('上课初期', 'opening').replace('上课关键期', 'efficient').replace('疲劳期', 'fatigue').replace('临近下课期', 'closing')
                analysis_results['segments'][seg_id] = {
                    'name': seg.get('name', ''),
                    'avg_effective_learning_rate': seg.get('avg_learning_rate', 0),
                    'avg_distraction_rate': seg.get('avg_distraction_rate', 0),
                    'avg_drowsiness_rate': 0,
                    'avg_positive_interaction_rate': 0,
                    'avg_misbehavior_rate': 0
                }
        
        # 生成报告
        try:
            report_result = generator.generate_report(analysis_results)
            report = report_result.get('report_content', '报告生成失败')
            # 确保报告不是失败状态
            if "失败" in report and len(report) < 50:
                report = generator._generate_mock_report(analysis_results)
        except Exception as report_error:
            print(f"报告生成失败，使用模拟报告: {report_error}")
            report = generator._generate_mock_report(analysis_results)
        
        # 保存报告文件
        report_filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # Step 5: 准备看板数据
        dashboard_data = {
            'video_name': video_name,
            'video_info': video_info,
            'stats': convert_to_json_serializable(stats),
            'segments': convert_to_json_serializable(segments),
            'report': report,
            'report_filename': report_filename,
            'data': convert_to_json_serializable(df.to_dict('records')),
            'columns': df.columns.tolist()
        }

        # 保存看板数据到JSON文件（供历史记录使用）
        video_name_for_file = video_name.replace('.mp4', '').replace('.avi', '').replace('.mov', '')
        dashboard_json_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{video_name_for_file}_dashboard.json')
        with open(dashboard_json_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

        return jsonify({
            'success': True,
            'message': '全流程分析完成！',
            'dashboard_data': dashboard_data,
            'video_name': video_name
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})

def calculate_segments(df):
    total_frames = len(df)
    if total_frames == 0:
        return []

    segment_duration = total_frames // 4 if total_frames >= 4 else 1

    total_col = 'total_students' if 'total_students' in df.columns else 'total_stu'

    segments = []
    for i in range(4):
        start = i * segment_duration
        end = min((i + 1) * segment_duration, total_frames)
        segment_df = df.iloc[start:end]

        if len(segment_df) > 0:
            segments.append({
                'name': ['上课初期', '上课关键期', '疲劳期', '临近下课期'][i],
                'start_frame': start,
                'end_frame': end,
                'avg_students': round(segment_df[total_col].mean(), 1),
                'avg_learning_rate': round(segment_df['effective_learning_rate'].mean(), 1),
                'avg_distraction_rate': round(segment_df['distraction_rate'].mean(), 1)
            })
    
    return segments

def calculate_decay_from_df(df):
    if len(df) < 2:
        return 0
    
    first_half = df.iloc[:len(df)//2]
    second_half = df.iloc[len(df)//2:]
    
    first_efficiency = first_half['effective_learning_rate'].mean()
    second_efficiency = second_half['effective_learning_rate'].mean()
    
    if first_efficiency == 0:
        return 0
    
    decay = ((first_efficiency - second_efficiency) / first_efficiency) * 100
    return round(decay, 1)

def load_config():
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {
        'sampling': {'frame_interval_sec': 3, 'max_frames': 900},
        'detection': {
            'model': 'yolov8l.pt',
            'conf_threshold': 0.15,
            'iou_threshold': 0.45,
            'min_bbox_area_ratio': 0.0015,
            'max_bbox_area_ratio': 0.4
        },
        'head_pose': {'pitch_up': 15, 'pitch_down': -10, 'yaw_side': 20},
        'output': {'cache_csv': 'outputs/csv/'}
    }

@app.route('/api/history')
def get_history():
    history = []
    if os.path.exists(app.config['OUTPUT_FOLDER']):
        for filename in sorted(os.listdir(app.config['OUTPUT_FOLDER']), reverse=True):
            if filename.endswith('_raw.csv'):
                filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                if os.path.isfile(filepath):
                    base_name = filename.replace('_raw.csv', '')
                    dashboard_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{base_name}_dashboard.json')
                    has_dashboard = os.path.exists(dashboard_path)
                    
                    # 检查报告文件是否存在
                    report_path = os.path.join(app.config['REPORTS_FOLDER'], f'analysis_report_{base_name}.txt')
                    has_report = os.path.exists(report_path)
                    
                    history.append({
                        'name': base_name,
                        'size': len(pd.read_csv(filepath)),
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S'),
                        'has_dashboard': has_dashboard,
                        'has_report': has_report
                    })
    return jsonify(history)

@app.route('/api/history/<filename>', methods=['DELETE'])
def delete_history(filename):
    try:
        # 提取基础文件名
        base_name = filename.replace('_raw.csv', '').replace('_dashboard.json', '')
        
        # 删除CSV文件
        csv_filename = f'{base_name}_raw.csv'
        csv_path = os.path.join(app.config['OUTPUT_FOLDER'], csv_filename)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        # 删除看板数据JSON文件
        dashboard_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{base_name}_dashboard.json')
        if os.path.exists(dashboard_path):
            os.remove(dashboard_path)
        
        # 删除报告文件
        report_path = os.path.join(app.config['REPORTS_FOLDER'], f'analysis_report_{base_name}.txt')
        if os.path.exists(report_path):
            os.remove(report_path)
        
        return jsonify({'success': True, 'message': '删除成功'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analyze_history/<filename>')
def analyze_history(filename):
    try:
        # 提取基础文件名
        if filename.endswith('_raw.csv'):
            base_name = filename.replace('_raw.csv', '')
        elif filename.endswith('_dashboard.json'):
            base_name = filename.replace('_dashboard.json', '')
        else:
            base_name = filename
        
        # 检查是否存在已保存的看板数据
        dashboard_json_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{base_name}_dashboard.json')
        if os.path.exists(dashboard_json_path):
            # 从已保存的JSON加载看板数据
            with open(dashboard_json_path, 'r', encoding='utf-8') as f:
                dashboard_data = json.load(f)
            return jsonify({
                'success': True,
                'dashboard_data': dashboard_data,
                'video_name': base_name
            })
        
        # 如果没有保存的看板数据，则重新生成
        csv_filename = f'{base_name}_raw.csv'
        csv_path = os.path.join(app.config['OUTPUT_FOLDER'], csv_filename)
        
        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'error': '文件不存在'})
        
        df = pd.read_csv(csv_path)
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

        # 计算总学生数（用于计算比率）
        total_students_sum = df[total_col].sum()
        
        # 计算参与率（专注听讲 + 低头学习 + 站立）
        focus_listen_sum = df.get('focus_listen', 0).sum() if 'focus_listen' in df.columns else 0
        study_bow_sum = df.get('study_bow', 0).sum() if 'study_bow' in df.columns else 0
        stand_up_sum = df.get('stand_up', 0).sum() if 'stand_up' in df.columns else 0
        engagement_rate = round((focus_listen_sum + study_bow_sum + stand_up_sum) / total_students_sum * 100, 1) if total_students_sum > 0 else 0
        
        # 计算分心率（放空 + 瞌睡 + 左顾右盼 + 松散坐姿）
        empty_mind_sum = df.get('empty_mind', 0).sum() if 'empty_mind' in df.columns else 0
        sleep_stu_sum = df.get('sleep_stu', 0).sum() if 'sleep_stu' in df.columns else 0
        look_side_sum = df.get('look_side', 0).sum() if 'look_side' in df.columns else 0
        loose_stu_sum = df.get('loose_stu', 0).sum() if 'loose_stu' in df.columns else 0
        distraction_rate = round((empty_mind_sum + sleep_stu_sum + look_side_sum + loose_stu_sum) / total_students_sum * 100, 1) if total_students_sum > 0 else 0
        
        stats = {
            'total_frames': len(df),
            'valid_frames': len(df),
            'total_students': total_students_sum,
            'avg_per_frame': round(total_students_sum / len(df), 1) if len(df) > 0 else 0,
            'avg_total_students': round(df[total_col].mean(), 1),
            'avg_effective_learning_rate': round(df['effective_learning_rate'].mean(), 1),
            'avg_distraction_rate': round(df['distraction_rate'].mean(), 1),
            'avg_drowsiness_rate': round(df['drowsiness_rate'].mean(), 1),
            'avg_positive_interaction_rate': round(df['interaction_rate'].mean(), 1),
            'avg_misbehavior_rate': round(df['misbehavior_rate'].mean(), 1),
            'attention_decay_rate': calculate_decay_from_df(df),
            'segments': segments,
            'engagement_rate': engagement_rate,
            'distraction_rate': distraction_rate,
            'sleep_count': int(sleep_stu_sum)
        }

        for col in behavior_cols:
            if col in df.columns:
                stats[col] = int(df[col].sum())
                stats[f'{col}_ratio'] = round(df[col].sum() / df[total_col].sum() * 100, 1) if df[total_col].sum() > 0 else 0

        # 添加 behaviors 结构以匹配前端期望
        stats['behaviors'] = {}
        for col in behavior_cols:
            if col in df.columns:
                count = int(df[col].sum())
                ratio = round(df[col].sum() / df[total_col].sum() * 100, 1) if df[total_col].sum() > 0 else 0
                stats['behaviors'][col] = {'count': count, 'ratio': ratio}
            else:
                stats['behaviors'][col] = {'count': 0, 'ratio': 0}

        # 重新生成报告（使用新的报告生成器）
        from src.analysis_module.report_generator import ReportGenerator
        api_key = "sk-02bf14d117fb415bbc28e7ce41a4c9db"
        generator = ReportGenerator(api_key=api_key)
        
        # 构建完整的分析结果数据
        analysis_results = {
            'overall': {
                'total_frames': stats.get('total_frames', 0),
                'avg_total_students': stats.get('avg_total_students', 0),
                'avg_effective_learning_rate': stats.get('avg_effective_learning_rate', 0),
                'avg_distraction_rate': stats.get('avg_distraction_rate', 0),
                'avg_drowsiness_rate': stats.get('avg_drowsiness_rate', 0),
                'avg_positive_interaction_rate': stats.get('avg_positive_interaction_rate', 0),
                'avg_misbehavior_rate': stats.get('avg_misbehavior_rate', 0),
                'max_effective_learning_rate': stats.get('avg_effective_learning_rate', 0),
                'min_effective_learning_rate': stats.get('avg_effective_learning_rate', 0)
            },
            'trends': {
                'attention_decay幅度': stats.get('attention_decay_rate', 0),
                'attention_decay_rate': stats.get('attention_decay_rate', 0),
                'learning_stability': 0,
                'misbehavior_increase': 0
            },
            'segments': {},
            'trend_analysis': {}
        }
        
        # 添加时段数据
        if stats.get('segments'):
            for seg in stats['segments']:
                seg_id = seg.get('name', '').replace('上课初期', 'opening').replace('上课关键期', 'efficient').replace('疲劳期', 'fatigue').replace('临近下课期', 'closing')
                analysis_results['segments'][seg_id] = {
                    'name': seg.get('name', ''),
                    'avg_effective_learning_rate': seg.get('avg_learning_rate', 0),
                    'avg_distraction_rate': seg.get('avg_distraction_rate', 0),
                    'avg_drowsiness_rate': 0,
                    'avg_positive_interaction_rate': 0,
                    'avg_misbehavior_rate': 0
                }
        
        # 生成报告
        try:
            report_result = generator.generate_report(analysis_results)
            report = report_result.get('report_content', '报告生成失败')
            if "失败" in report and len(report) < 50:
                report = generator._generate_mock_report(analysis_results)
        except Exception as report_error:
            print(f"报告生成失败，使用模拟报告: {report_error}")
            report = generator._generate_mock_report(analysis_results)
        
        # 使用视频名作为报告文件名的一部分
        report_filename = f"analysis_report_{base_name}.txt"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # 将DataFrame转换为列表供前端使用
        df_records = convert_to_json_serializable(df.to_dict('records'))
        
        dashboard_data = {
            'video_name': base_name,
            'video_info': {},
            'stats': convert_to_json_serializable(stats),
            'segments': convert_to_json_serializable(segments),
            'report': report,
            'report_filename': report_filename,
            'data': df_records,
            'columns': df.columns.tolist()
        }

        # 保存看板数据到JSON文件（供历史记录使用）
        dashboard_json_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{base_name}_dashboard.json')
        with open(dashboard_json_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

        return jsonify({
            'success': True,
            'dashboard_data': dashboard_data,
            'video_name': base_name
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})

@app.route('/reports/<filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
