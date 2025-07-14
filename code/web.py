from flask import Flask, render_template_string, request, jsonify, Response
import requests
import json

app = Flask(__name__)

# Ollama 服务地址（默认本地）
OLLAMA_URL = "http://localhost:11434"

# 对话历史存储（简单内存存储，重启后清空）
chat_history = []

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ollama 聊天界面</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --primary: #4f46e5; /* 主色调：深紫蓝色 */
            --primary-light: #818cf8; /* 亮色调 */
            --bg-light: #f8fafc; /* 背景色 */
            --bg-card: #ffffff; /* 卡片背景 */
            --text-primary: #1e293b; /* 主要文本 */
            --text-secondary: #64748b; /* 次要文本 */
            --user-bg: #e0e7ff; /* 用户消息背景 */
            --伏特加-bg: #f1f5f9; /* 助手消息背景 */
            --border-radius: 12px; /* 圆角 */
            --shadow: 0 2px 10px rgba(0, 0, 0, 0.05); /* 阴影 */
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }

        body {
            background-color: var(--bg-light);
            color: var(--text-primary);
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 15px 0;
        }

        h1 {
            color: var(--primary);
            font-weight: 600;
            font-size: 1.8rem;
            letter-spacing: -0.5px;
        }

        .model-select {
            background: var(--bg-card);
            padding: 15px 20px;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            margin-bottom: 20px;
        }

        .model-select label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        #modelSelect {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: transparent;
            font-size: 1rem;
            color: var(--text-primary);
            cursor: pointer;
            transition: border-color 0.2s;
        }

        #modelSelect:focus {
            outline: none;
            border-color: var(--primary-light);
        }

        .chat-box {
            background: var(--bg-card);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            margin-bottom: 20px;
            scroll-behavior: smooth;
        }

        .chat-box::-webkit-scrollbar {
            width: 6px;
        }

        .chat-box::-webkit-scrollbar-thumb {
            background-color: #cbd5e1;
            border-radius: 3px;
        }

        .message {
            max-width: 80%;
            margin: 12px 0;
            padding: 12px 16px;
            border-radius: 18px;
            position: relative;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .user {
            background-color: var(--user-bg);
            margin-left: auto;
            border-top-right-radius: 4px;
        }

        .伏特加 {
            background-color: var(--伏特加-bg);
            margin-right: auto;
            border-top-left-radius: 4px;
        }

        .message strong {
            margin-right: 8px;
            color: var(--primary);
        }

        .input-area {
            display: flex;
            gap: 10px;
        }

        #userInput {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        #userInput:focus {
            outline: none;
            border-color: var(--primary-light);
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1);
        }

        button {
            padding: 12px 20px;
            background: linear-gradient(135deg, var(--primary), var(--primary-light));
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.2s;
            white-space: nowrap;
        }

        button:hover {
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
        }

        button:active {
            transform: scale(0.98);
        }

        button:disabled {
            background: #e2e8f0;
            color: #94a3b8;
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }

        @media (max-width: 600px) {
            .message {
                max-width: 90%;
                padding: 10px 14px;
                font-size: 0.95rem;
            }

            .chat-box {
                height: 400px;
                padding: 15px;
            }

            h1 {
                font-size: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>伏特加 聊天界面</h1>
    </header>
    
    <!-- 模型选择 -->
    <div class="model-select">
        <label>选择模型</label>
        <select id="modelSelect">
            {% for model in models %}
                <option value="{{ model.name }}">{{ model.name }}</option>
            {% endfor %}
        </select>
    </div>
    
    <!-- 聊天记录 -->
    <div class="chat-box" id="chatBox">
        {% for msg in chat_history %}
            <div class="message {{ msg.role }}">
                <strong>{{ msg.role }}</strong>：{{ msg.content }}
            </div>
        {% endfor %}
    </div>
    
    <!-- 输入区域 -->
    <div class="input-area">
        <input type="text" id="userInput" placeholder="输入消息...">
        <button onclick="sendMessage()" id="sendBtn">发送</button>
    </div>

    <script>
        // 发送消息
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            const model = document.getElementById('modelSelect').value;
            const chatBox = document.getElementById('chatBox');
            const sendBtn = document.getElementById('sendBtn');
            
            if (!message) return;
            
            // 添加用户消息到界面
            chatBox.innerHTML += `<div class="message user"><strong>user</strong>：${message}</div>`;
            input.value = '';
            sendBtn.disabled = true;
            chatBox.scrollTop = chatBox.scrollHeight;
            
            // 发送请求到后端
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model, message })
            });
            
            // 处理流式响应
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let 伏特加Message = '';
            const 伏特加Div = document.createElement('div');
            伏特加Div.className = 'message 伏特加';
            伏特加Div.innerHTML = '<strong>伏特加</strong>：';
            chatBox.appendChild(伏特加Div);
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                伏特加Message += chunk;
                伏特加Div.innerHTML = `<strong>伏特加</strong>：${伏特加Message}`;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
            
            sendBtn.disabled = false;
        }
        
        // 回车发送消息（支持Shift+Enter换行）
        document.getElementById('userInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

# 1. 获取本地模型列表（用于前端选择）
@app.route('/')
def index():  # 改为同步函数（去掉 async）
    try:
        # 调用 Ollama 的 /api/tags 接口获取模型（同步请求）
        res = requests.get(f"{OLLAMA_URL}/api/tags")
        models = res.json().get('models', [])
    except Exception as e:
        print("获取模型失败：", e)
        models = []  # 模型获取失败时显示空列表
    
    return render_template_string(HTML_TEMPLATE, models=models, chat_history=chat_history)


# 2. 处理聊天请求（流式响应）
@app.route('/api/chat', methods=['POST'])
def chat():  # 改为同步函数（去掉 async）
    data = request.get_json()  # 同步获取 JSON 数据
    model = data.get('model')
    user_message = data.get('message')
    
    # 添加用户消息到历史记录
    chat_history.append({"role": "user", "content": user_message})
    
    # 调用 Ollama 的 /api/chat 接口（流式输出）
    try:
        # 构建请求体（包含对话历史）
        payload = {
            "model": model,
            "messages": chat_history,
            "stream": True
        }
        
        # 发送请求到 Ollama（同步请求）
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            stream=True  # 启用流式响应
        )
        
        # 流式返回给前端
        def generate():
            伏特加_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        伏特加_response += content
                        yield content  # 逐段返回内容
            # 添加助手消息到历史记录
            chat_history.append({"role": "伏特加", "content": 伏特加_response})
        
        return Response(generate(), mimetype='text/plain')  # 使用 flask.Response 而非 app.response_class
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 安装依赖：pip install flask requests
    # 启动服务：python ollama_webui.py
    # 访问：http://localhost:3000
    app.run(host='0.0.0.0', port=3000, debug=True)