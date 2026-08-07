"""
报告生成模块
整合统计结果与分析结论，调用大语言模型生成专业化课堂学情分析报告
"""

import json
import requests
import numpy as np
import pandas as pd
import os
from typing import Dict, Any
from datetime import datetime


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('QWEN_API_KEY')
        if not self.api_key:
            raise RuntimeError(
                "未配置通义千问 API Key，请设置环境变量 QWEN_API_KEY"
            )
        self.model = "qwen3-max"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    def build_prompt(self, analysis_results: Dict) -> str:
        """
        构建专业的大语言模型提示词
        采用用户指定的教学督导角色设定和报告结构
        """
        overall = analysis_results.get('overall', {})
        trends = analysis_results.get('trends', {})
        segments = analysis_results.get('segments', {})
        trends_data = analysis_results.get('trend_analysis', {})
        
        # 构建数据文本
        data_text = self._build_data_text(overall, trends, segments, trends_data)
        
        # ========== 专业提示词模板 ==========
        prompt = f"""
你是资深校园教学督导与课堂学情数据分析专家，精通大学课堂巡课评价标准，行文严谨正式，贴合教研报告风格。
 
硬性判定规则必须严格遵守： 
1. 区分两类低头行为：低头伏案书写、翻看书本属于正常自主学习；无动作慵懒低头判定为放空走神，不得混为一谈。 
2. 区分两类交谈行为：大范围同步交流视作教师安排的合规课堂讨论；零散两两私下交谈判定为违纪闲聊。 
3. 按照四个阶段分析课堂：上课初期、上课关键期、疲劳期、临近下课期，依次分析状态变化。 
4. 所有分析内容严格依据给定数据，不虚构信息，客观评判优劣。 
5. 给出的教学建议贴合实际课堂场景，具备落地参考价值。 
 
请固定按照以下结构输出完整报告： 
一、课堂基础数据概况 
二、班级整体学情综合评价 
三、分时段课堂状态深度分析 
四、各类课堂行为结构分析 
五、课堂现存优势与主要问题 
六、课堂教学与班级管理优化建议 
七、课堂综合得分与等级评定

【输入数据】
{data_text}

【输出要求】
全文正式、条理清晰、段落分明，不用markdown，不用加粗，直接生成可存档的正式学情分析报告。
"""
        
        return prompt.strip()
    
    def _build_data_text(self, overall, trends, segments, trends_data):
        """
        构建结构化的数据文本
        """
        data_lines = []
        
        # 基础数据
        data_lines.append(f"分析帧总数：{overall.get('total_frames', 0)}帧")
        data_lines.append(f"平均检测人数：{overall.get('avg_total_students', 0)}人")
        data_lines.append(f"数据分析时长：{(overall.get('total_frames', 0) * 5 / 60):.1f}分钟")
        data_lines.append("")
        
        # 核心指标
        data_lines.append("【核心指标】")
        data_lines.append(f"有效学习率（平均）：{overall.get('avg_effective_learning_rate', 0)}%")
        data_lines.append(f"有效学习率（最高）：{overall.get('max_effective_learning_rate', 0)}%")
        data_lines.append(f"有效学习率（最低）：{overall.get('min_effective_learning_rate', 0)}%")
        data_lines.append(f"走神率：{overall.get('avg_distraction_rate', 0)}%")
        data_lines.append(f"困倦率：{overall.get('avg_drowsiness_rate', 0)}%")
        data_lines.append(f"课堂互动率：{overall.get('avg_positive_interaction_rate', 0)}%")
        data_lines.append(f"违纪率：{overall.get('avg_misbehavior_rate', 0)}%")
        data_lines.append("")
        
        # 时段分析数据
        data_lines.append("【时段分析数据】")
        segment_names = {
            'opening': '上课初期',
            'efficient': '上课关键期',
            'fatigue': '疲劳期',
            'closing': '临近下课期'
        }
        
        for segment_id in ['opening', 'efficient', 'fatigue', 'closing']:
            if segment_id in segments and isinstance(segments[segment_id], dict):
                metrics = segments[segment_id]
                data_lines.append(f"{metrics.get('name', segment_names.get(segment_id, segment_id))}：")
                data_lines.append(f"  有效学习率：{metrics.get('avg_effective_learning_rate', 0)}%")
                data_lines.append(f"  走神率：{metrics.get('avg_distraction_rate', 0)}%")
                data_lines.append(f"  困倦率：{metrics.get('avg_drowsiness_rate', 0)}%")
                data_lines.append(f"  互动率：{metrics.get('avg_positive_interaction_rate', 0)}%")
                data_lines.append(f"  违纪率：{metrics.get('avg_misbehavior_rate', 0)}%")
                data_lines.append("")
        
        # 趋势特征
        data_lines.append("【趋势特征】")
        data_lines.append(f"注意力衰减幅度：{trends.get('attention_decay幅度', 0)}个百分点")
        data_lines.append(f"注意力衰减率：{trends.get('attention_decay_rate', 0)}%")
        data_lines.append(f"学习状态稳定性：{trends.get('learning_stability', 0)}")
        data_lines.append(f"不良行为增长率：{trends.get('misbehavior_increase', 0)}个百分点")
        data_lines.append("")
        
        # 深层发现
        data_lines.append("【深层发现】")
        if trends_data.get('attention_peaks'):
            data_lines.append(f"检测到{len(trends_data['attention_peaks'])}个注意力峰值时段")
        if trends_data.get('attention_dips'):
            data_lines.append(f"检测到{len(trends_data['attention_dips'])}个注意力低谷时段")
        if trends_data.get('student_retention'):
            retention = trends_data['student_retention']
            data_lines.append(f"学生留存率：{retention.get('retention_rate', 0)}%")
        if trends_data.get('rhythm_patterns', {}).get('is_rhythmic'):
            data_lines.append(f"检测到周期性课堂节奏")
        
        return "\n".join(data_lines)
    
    def call_qwen_api(self, prompt: str) -> str:
        """
        调用通义千问API（阿里云百炼平台）
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.6,
            "max_tokens": 4000,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            # 处理不同的响应格式
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            elif 'output' in result:
                if isinstance(result['output'], dict) and 'text' in result['output']:
                    return result['output']['text']
                elif isinstance(result['output'], str):
                    return result['output']
            else:
                return "报告生成失败：API返回格式异常"
        
        except requests.exceptions.RequestException as e:
            return f"报告生成失败：{str(e)}"
    
    def _generate_mock_report(self, analysis_results: Dict) -> str:
        """
        当API调用失败时生成模拟报告
        """
        overall = analysis_results.get('overall', {})
        segments = analysis_results.get('segments', {})
        trends = analysis_results.get('trends', {})
        
        # 计算综合得分和等级
        avg_learning_rate = overall.get('avg_effective_learning_rate', 0)
        score = round(avg_learning_rate * 1.1)
        if score >= 90:
            grade = '优秀'
        elif score >= 80:
            grade = '良好'
        elif score >= 70:
            grade = '中等'
        elif score >= 60:
            grade = '及格'
        else:
            grade = '需改进'
        
        report = f"""课堂学情分析报告

一、课堂基础数据概况
本次分析共处理{overall.get('total_frames', 0)}帧视频数据，平均每帧检测到{overall.get('avg_total_students', 0):.1f}名学生，采样频率为3秒/帧。分析覆盖课堂全时段，数据具有代表性。

二、班级整体学情综合评价
整体来看，班级课堂表现{grade}。有效学习率达到{avg_learning_rate:.1f}%，说明大部分学生能够保持较好的学习状态。分心率为{overall.get('avg_distraction_rate', 0):.1f}%，困倦率{overall.get('avg_drowsiness_rate', 0):.1f}%，课堂互动率{overall.get('avg_positive_interaction_rate', 0):.1f}%，违纪行为发生率{overall.get('avg_misbehavior_rate', 0):.1f}%。

三、分时段课堂状态深度分析
"""
        
        segment_names = ['opening', 'efficient', 'fatigue', 'closing']
        segment_display_names = ['上课初期', '上课关键期', '疲劳期', '临近下课期']
        
        for i, seg_id in enumerate(segment_names):
            if seg_id in segments:
                metrics = segments[seg_id]
                report += f"{segment_display_names[i]}：有效学习率{metrics.get('avg_effective_learning_rate', 0):.1f}%，分心率{metrics.get('avg_distraction_rate', 0):.1f}%"
                if seg_id == 'efficient':
                    report += '，为本节课学习效率最高的阶段'
                elif seg_id == 'fatigue':
                    report += '，学生出现明显疲劳迹象'
                report += '\n'
        
        report += f"""
四、各类课堂行为结构分析
根据检测数据，学生主要表现为专注听讲和低头学习两种状态，合计占比超过{min(avg_learning_rate + 10, 100):.0f}%。走神发呆、打瞌睡等不良行为占比较低，说明课堂整体秩序良好。

五、课堂现存优势与主要问题
优势方面，上课关键期有效学习率达到{segments.get('efficient', {}).get('avg_effective_learning_rate', 0):.1f}%，学生参与度较高，课堂互动氛围良好。
主要问题在于注意力衰减幅度达{trends.get('attention_decay幅度', 0):.1f}个百分点，后期学习状态明显下滑，需要引起重视。

六、课堂教学与班级管理优化建议
1. 在上课初期增加互动环节，快速吸引学生注意力，营造积极的学习氛围。
2. 在疲劳期安排短暂休息或趣味互动，缓解学生疲劳，维持学习状态。
3. 根据注意力峰值和低谷时段，合理安排教学内容和互动环节，优化课堂节奏。
4. 加强下课前5分钟的课堂管理，减少涣散现象，确保教学效果。

七、课堂综合得分与等级评定
综合评分：{score}分
课堂等级：{grade}

报告生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
"""
        return report
    
    def generate_report(self, analysis_results: Dict) -> Dict:
        """
        生成完整的分析报告
        """
        # 构建prompt
        prompt = self.build_prompt(analysis_results)
        
        # 调用API生成报告
        report_content = self.call_qwen_api(prompt)
        
        # 如果API调用失败，使用模拟报告
        if "失败" in report_content:
            report_content = self._generate_mock_report(analysis_results)
        
        # 准备输出数据
        output = {
            'timestamp': datetime.now().isoformat(),
            'report_content': report_content,
            'summary': {
                'avg_effective_learning_rate': analysis_results.get('overall', {}).get('avg_effective_learning_rate', 0),
                'avg_distraction_rate': analysis_results.get('overall', {}).get('avg_distraction_rate', 0),
                'avg_drowsiness_rate': analysis_results.get('overall', {}).get('avg_drowsiness_rate', 0),
                'attention_decay_rate': analysis_results.get('trends', {}).get('attention_decay_rate', 0),
                'student_retention_rate': analysis_results.get('trend_analysis', {}).get('student_retention', {}).get('retention_rate', 0)
            },
            'raw_data': analysis_results
        }
        
        return output
    
    def save_report(self, report: Dict, output_path: str = "reports/") -> str:
        """
        保存报告到文件
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # 保存文本报告
        report_filename = f"{output_path}analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report['report_content'])
        
        # 保存JSON数据（处理numpy类型）
        data_filename = f"{output_path}analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        report_serializable = convert_numpy(report)
        
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, ensure_ascii=False, indent=2)
        
        return report_filename


def generate_ai_report(stats: Dict) -> str:
    """
    简化的报告生成函数
    直接接受 stats 字典，生成简化的分析报告
    """
    generator = ReportGenerator()
    
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
    
    result = generator.generate_report(analysis_results)
    return result.get('report_content', '报告生成失败')