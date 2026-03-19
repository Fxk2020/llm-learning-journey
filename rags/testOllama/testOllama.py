from ollama import chat

stream = chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': '你有思考的能力吗?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)