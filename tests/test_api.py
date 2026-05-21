"""
测试阿里云百炼平台API - 使用正确的OpenAI兼容格式
"""

import requests

def test_dashscope_openai_api():
    api_key = "sk-02bf14d117fb415bbc28e7ce41a4c9db"
    model = "qwen-plus-2025-07-28"
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "你好，请用中文简单介绍一下自己"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    try:
        print(f"测试模型: {model}")
        print(f"API地址: {url}")
        print("正在发送请求...")
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ API调用成功！")
            print(f"响应结构: {result.keys()}")
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"\n回复内容:\n{content}")
            else:
                print(f"完整响应: {result}")
                
        else:
            print(f"\n❌ API调用失败")
            try:
                error_info = response.json()
                print(f"错误信息: {error_info}")
            except:
                print(f"响应内容: {response.text[:500]}")
                
    except Exception as e:
        print(f"\n❌ 异常错误: {str(e)}")

if __name__ == "__main__":
    test_dashscope_openai_api()