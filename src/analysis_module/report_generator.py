"""
报告生成模块
整合统计结果与分析结论，调用大语言模型生成专业化课堂学情分析报告
"""

import json
import requests
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "sk-02bf14d117fb415bbc28e7ce41a4c9db"
        self.model = "qwen-plus-2025-07-28"
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
【角色设定】
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
            'opening': '开课适应期',
            'efficient': '高效学习期',
            'fatigue': '疲劳下滑期',
            'closing': '下课涣散期'
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
        
        report = f"""【课堂学情分析报告】

一、课堂基础数据概况
- 总帧数：{overall.get('total_frames', 0)}帧
- 平均检测人数：{overall.get('avg_total_students', 0):.1f}人
- 采样频率：5秒/帧

二、班级整体学情综合评价
- 有效学习率：{overall.get('avg_effective_learning_rate', 0):.1f}%
- 走神率：{overall.get('avg_distraction_rate', 0):.1f}%
- 困倦率：{overall.get('avg_drowsiness_rate', 0):.1f}%
- 互动率：{overall.get('avg_positive_interaction_rate', 0):.1f}%
- 违纪率：{overall.get('avg_misbehavior_rate', 0):.1f}%

三、四大授课时段学情分析
"""
        for seg_id, metrics in segments.items():
            report += f"- {metrics['name']}：有效学习率{metrics.get('avg_effective_learning_rate', 0):.1f}%，共{metrics.get('frame_count', 0)}帧\n"
        
        report += f"""
四、课堂现存优势与突出问题
- 优势：高效学习期有效学习率达到{segments.get('efficient', {}).get('avg_effective_learning_rate', 0):.1f}%，学生参与度较高
- 问题：注意力衰减幅度达{trends.get('attention_decay幅度', 0):.1f}个百分点，后期学习状态明显下滑

五、教学优化与班级管理建议
1. 在开课适应期增加互动环节，快速吸引学生注意力
2. 在疲劳下滑期安排短暂休息或趣味互动
3. 加强下课前5分钟的课堂管理，减少涣散现象

六、综合评分与课堂等级评定
- 综合评分：{overall.get('avg_effective_learning_rate', 0) * 1.1:.1f}分
- 课堂等级：{'优秀' if overall.get('avg_effective_learning_rate', 0) >= 70 else '良好' if overall.get('avg_effective_learning_rate', 0) >= 50 else '需改进'}

【报告说明】
本报告由系统自动生成，若需更详细的分析建议，请确保API配置正确后重新生成。
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