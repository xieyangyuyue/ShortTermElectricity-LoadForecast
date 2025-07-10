import requests
import json
import time
from typing import Dict, Any, Generator, Optional

class OllamaClient:
    """Ollama API客户端，用于与本地运行的Ollama服务进行交互"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        初始化Ollama API客户端
        
        Args:
            base_url: Ollama服务的基础URL，默认为http://localhost:11434
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate(self, 
                 model: str, 
                 prompt: str, 
                 system: Optional[str] = None,
                 context: Optional[list] = None,
                 stream: bool = True,
                 **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        调用Ollama模型生成文本
        
        Args:
            model: 要使用的模型名称
            prompt: 输入的提示文本
            system: 系统指令，用于设置模型行为
            context: 对话上下文，用于持续对话
            stream: 是否使用流式响应
            **kwargs: 其他可选参数，如temperature、top_p等
            
        Yields:
            生成的文本块（流式响应）或完整响应（非流式）
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        
        # 添加可选参数
        if system is not None:
            payload["system"] = system
        if context is not None:
            payload["context"] = context
        payload.update(kwargs)
        
        try:
            # 发送请求
            response = self.session.post(url, json=payload, stream=stream)
            response.raise_for_status()  # 检查HTTP状态码
            
            if stream:
                # 处理流式响应
                for line in response.iter_lines():
                    if line:
                        # 解码并解析JSON
                        chunk = json.loads(line.decode('utf-8'))
                        yield chunk
            else:
                # 处理非流式响应
                yield response.json()
                
        except requests.exceptions.RequestException as e:
            print(f"API请求错误: {e}")
            yield {"error": str(e)}
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            yield {"error": f"响应解析错误: {e}"}
    
    def chat(self, model: str, messages: list, stream: bool = True, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        进行多轮对话
        
        Args:
            model: 要使用的模型名称
            messages: 对话消息列表，每个消息是一个字典，包含"role"和"content"
            stream: 是否使用流式响应
            **kwargs: 其他可选参数
            
        Yields:
            生成的回复消息块
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        payload.update(kwargs)
        
        try:
            response = self.session.post(url, json=payload, stream=stream)
            response.raise_for_status()
            
            if stream:
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line.decode('utf-8'))
            else:
                yield response.json()
                
        except Exception as e:
            print(f"聊天请求错误: {e}")
            yield {"error": str(e)}
    
    def list_models(self) -> Dict[str, Any]:
        """获取可用模型列表"""
        url = f"{self.base_url}/api/tags"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取模型列表错误: {e}")
            return {"error": str(e), "models": []}
    
    def pull_model(self, name: str, stream: bool = True, max_retries: int = 3) -> Generator[Dict[str, Any], None, None]:
        """
        拉取模型（如果本地不存在）
        
        Args:
            name: 模型名称
            stream: 是否使用流式响应显示下载进度
            max_retries: 最大重试次数
            
        Yields:
            下载进度信息
        """
        url = f"{self.base_url}/api/pull"
        payload = {"name": name, "stream": stream}
        
        retries = 0
        while retries < max_retries:
            try:
                response = self.session.post(url, json=payload, stream=stream)
                response.raise_for_status()
                
                if stream:
                    for line in response.iter_lines():
                        if line:
                            yield json.loads(line.decode('utf-8'))
                else:
                    yield response.json()
                    
                # 下载成功，退出重试循环
                break
                    
            except Exception as e:
                retries += 1
                print(f"\n下载中断 ({retries}/{max_retries}): {e}")
                if retries < max_retries:
                    print("正在重试...")
                    # 等待几秒再重试
                    time.sleep(2)
                else:
                    print("达到最大重试次数，下载失败")
                    yield {"error": str(e), "retries_exceeded": True}


def main():
    """演示如何使用OllamaClient"""
    client = OllamaClient()
    
    # 列出可用模型
    print("可用模型:")
    models = client.list_models()
    for model in models.get('models', []):
        print(f"- {model['name']}")
    
    # 使用现有模型而非下载新模型
    model_to_use = "deepseek-r1:7b"  # 直接指定使用已有模型
    print(f"\n将使用模型: {model_to_use}")
    
    # 生成文本示例
    print(f"\n使用 {model_to_use} 生成文本示例:")
    prompt = "请介绍一下人工智能"
    print(f"提示: {prompt}")
    print("回答: ", end='')
    
    full_response = ""
    for chunk in client.generate(model=model_to_use, prompt=prompt, stream=True):
        if 'response' in chunk:
            print(chunk['response'], end='', flush=True)
            full_response += chunk['response']
    print()
    
    # 聊天示例
    print("\n聊天示例:")
    messages = [
        {"role": "user", "content": "什么是机器学习?"},
    ]
    
    print(f"用户: {messages[0]['content']}")
    print("AI: ", end='')
    
    full_chat_response = ""
    for chunk in client.chat(model=model_to_use, messages=messages, stream=True):
        if 'message' in chunk and 'content' in chunk['message']:
            print(chunk['message']['content'], end='', flush=True)
            full_chat_response += chunk['message']['content']
    print()
    
    # 添加AI回复到对话历史
    messages.append({"role": "assistant", "content": full_chat_response})
    
    # 继续对话
    follow_up = "它与深度学习有什么区别?"
    messages.append({"role": "user", "content": follow_up})
    
    print(f"\n用户: {follow_up}")
    print("AI: ", end='')
    
    full_chat_response = ""
    for chunk in client.chat(model=model_to_use, messages=messages, stream=True):
        if 'message' in chunk and 'content' in chunk['message']:
            print(chunk['message']['content'], end='', flush=True)
            full_chat_response += chunk['message']['content']
    print()

if __name__ == "__main__":
    main()