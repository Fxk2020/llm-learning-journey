# Use a pipeline as a high-level helper
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 设置你的 HuggingFace Token（从 https://huggingface.co/settings/tokens 获取）
api_key = os.getenv('HF_TOKEN')

from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
messages = [
    {"role": "user", "content": "Who are you?"},
]
print(pipe(messages))
