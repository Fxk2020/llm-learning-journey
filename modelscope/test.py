import dashscope
import os
from dashscope.api_entities.dashscope_response import HTTPStatus

# 替换为你那个“报错”的 Key

api_key = os.getenv('api_key')

def test_key():
    try:
        response = dashscope.Generation.call(
            model='qwen-turbo',
            api_key=api_key,
            prompt='1+1等于几？'
        )
        if response.status_code == HTTPStatus.OK:
            print("✅ 破案了：Key 是完全正常的！是 Dify 那边抽风了。")
            print("回答：", response.output.text)
        else:
            print("❌ 破案了：Key 确实失效了。")
            print(f"错误码: {response.code}")
            print(f"错误信息: {response.message}")
            if "PaymentRequired" in str(response):
                print("👉 原因：欠费了！快去充值！")
            elif "Throttling" in str(response):
                print("👉 原因：请求太快被限流了，歇一会就好。")
    except Exception as e:
        print(f"❌ 调用失败: {e}")

if __name__ == '__main__':
    test_key()