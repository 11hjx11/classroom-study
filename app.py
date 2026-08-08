from flask import Flask, render_template, request, jsonify
import os
import sys
import json

app = Flask(__name__,
            template_folder='frontend/templates',
            static_folder='frontend/static')

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'cache_csv'
app.config['REPORTS_FOLDER'] = 'outputs/reports'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============ Agent 核心逻辑 ============

_agent_instance = None
_agent_mode = "single"  # "single" 或 "multi"

def get_agent():
    global _agent_instance, _agent_mode
    if _agent_instance is None:
        from src.agents.orchestrator import ClassAgent
        _agent_instance = ClassAgent()
    return _agent_instance

def get_multi_agent():
    """获取多 Agent 实例（懒加载）"""
    global _agent_instance
    if _agent_instance is None or not hasattr(_agent_instance, '_sub_agent_tools'):
        from src.agents.multi_agent import MultiAgentOrchestrator
        _agent_instance = MultiAgentOrchestrator()
    return _agent_instance

def set_agent_mode(mode: str):
    """切换 Agent 模式"""
    global _agent_mode, _agent_instance
    _agent_mode = mode
    _agent_instance = None  # 重置实例，下次 get_agent 时重建

# ============ 路由 ============

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'success': False, 'error': '消息不能为空'})

        if _agent_mode == "multi":
            agent = get_multi_agent()
        else:
            agent = get_agent()
        result = agent.chat(user_message)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})

@app.route('/api/chat_stream', methods=['POST'])
def chat_stream():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'success': False, 'error': '消息不能为空'})

        if _agent_mode == "multi":
            agent = get_multi_agent()
        else:
            agent = get_agent()

        def generate():
            try:
                for event in agent.chat_stream(user_message):
                    sse_data = json.dumps({
                        'type': event.get('type', 'message'),
                        'data': event.get('data', ''),
                        'message': event.get('message', ''),
                    }, ensure_ascii=False, default=str)
                    yield f"data: {sse_data}\n\n"
            except Exception as e:
                err_data = json.dumps({
                    'type': 'error',
                    'data': None,
                    'message': str(e),
                }, ensure_ascii=False)
                yield f"data: {err_data}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return app.response_class(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*',
            }
        )
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})

@app.route('/api/chat/reset', methods=['POST'])
def chat_reset():
    try:
        agent = get_agent()
        agent.reset()
        return jsonify({'success': True, 'message': '对话已重置'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chat/tools')
def chat_tools():
    try:
        agent = get_agent()
        schemas = agent.get_available_tools()
        return jsonify({'success': True, 'tools': schemas})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chat/greeting')
def chat_greeting():
    try:
        agent = get_agent()
        return jsonify({'success': True, 'greeting': agent.get_greeting()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/agent/mode', methods=['GET', 'POST'])
def agent_mode():
    """获取或切换 Agent 模式（single/multi）"""
    global _agent_mode
    if request.method == 'POST':
        data = request.get_json() or {}
        mode = data.get('mode', 'single')
        if mode in ('single', 'multi'):
            set_agent_mode(mode)
            return jsonify({'success': True, 'mode': mode, 'message': f'已切换到{"单" if mode=="single" else "多"} Agent 模式'})
        return jsonify({'success': False, 'error': '无效的模式，可选: single, multi'})
    return jsonify({'success': True, 'mode': _agent_mode})

@app.route('/api/agent/usage')
def agent_usage():
    """获取上次对话的 Token 用量"""
    try:
        agent = get_agent()
        return jsonify({'success': True, 'usage': agent.get_last_usage()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
