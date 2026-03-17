# my_ai_clients.py
import requests
import json
import re

# 本地 Ollama 服务地址（默认）
OLLAMA_URL = "http://localhost:11434/api/generate"

def call_ollama(model_name, prompt):
    """
    通用函数：向本地 Ollama 服务发送请求，生成回复。
    :param model_name: 本地已下载的模型名称，如 'deepseek-r1:7b'
    :param prompt: 输入提示词
    :return: 模型生成的文本，失败返回 None
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,          # 关闭流式输出，一次性获取完整回复
        "options": {
            "temperature": 0.7,    # 控制随机性，保持与之前一致
            "num_predict": 800      # 相当于 max_tokens
        }
    }
    try:
        # 设置较长的超时时间，因为本地模型推理可能较慢
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        # Ollama 的响应中，生成的文本位于 'response' 字段
        return result.get('response', '')
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到 Ollama 服务。请确认 Ollama 是否已启动（运行 'ollama serve'）。")
        return None
    except Exception as e:
        print(f"❌ 调用本地模型 {model_name} 时出错: {e}")
        return None
    

def generate_clinical_advice(ai_model, diagnosis_label, probability, age, gender):
    """
    根据前端选择的 AI 模型，调用对应的本地 Ollama 模型生成个性化临床建议。
    :param ai_model: 前端传入的模型标识，如 'deepseek', 'doubao', 'kimi'
    :param diagnosis_label: 诊断标签，如 'Demented'
    :param probability: 置信度 (0-1)
    :param age: 年龄
    :param gender: 性别字符串 'male' 或 'female'
    :return: 生成的建议文本，失败返回 None
    """
    # 构建提示词（保持原样，无需修改）
    prompt = f"""
你是一位经验丰富的神经科专家，请根据以下患者信息生成一份专业、个性化的临床建议。

患者信息：
- 年龄：{age}岁
- 性别：{'男' if gender.lower() in ['male', 'm'] else '女'}
- 诊断结果：{diagnosis_label}（{'痴呆' if diagnosis_label=='Demented' else '正常' if diagnosis_label=='Nondemented' else '转化期'}）
- 诊断概率：{probability:.1%}

请用中文生成一份完整的建议，包括以下部分：
1. 当前状况解读（基于概率和诊断）
2. 具体建议（药物治疗、生活方式调整、必要检查等）
3. 随访计划（建议复查时间）
4. 对家属的建议（如何照顾、注意事项）

要求：语言亲切、专业，针对患者的具体情况提供个性化指导。
"""

    # 根据前端选项映射到本地模型名称
    if ai_model == "deepseek":
        model = "deepseek-r1:1.5b"
        print(f"🧠 调用本地模型: {model}")
    elif ai_model == "qwen2.5":
        model = "qwen2.5:3b"          # 用通义千问 2.5 替代豆包
        print(f"🧠 调用本地模型: {model}")
    elif ai_model == "llama3.2":
        model = "llama3.2:3b"          # 用 Llama 3.2 替代 Kimi
        print(f"🧠 调用本地模型: {model}")
    else:
        return None

    # 调用本地模型
    response = call_ollama(model, prompt)

    # ---------- 新增后处理 ----------
    if response:
        # 去除文本开头可能存在的空白（包括换行、空格等）
        response = re.sub(r'^[\s\u00A0\u200b\u200c\u200d\uFEFF]+', '', response)

        # 将“标题：”后紧跟的两个换行符替换为一个换行符
        response = re.sub(r'([^：]+：)\n\n', r'\1\n', response)

    return response