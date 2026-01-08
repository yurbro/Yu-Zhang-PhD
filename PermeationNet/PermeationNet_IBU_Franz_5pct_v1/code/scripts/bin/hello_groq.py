# import os
# from openai import OpenAI

# client = OpenAI(
#     api_key=os.environ["GROQ_API_KEY"],  # ✅ 读环境变量名
#     base_url="https://api.groq.com/openai/v1",
# )

# resp = client.chat.completions.create(
#     model="llama-3.1-8b-instant",
#     messages=[{"role": "user", "content": "Say ping and nothing else."}],
#     temperature=0.2,
# )

# print(resp.choices[0].message.content)

# import os

# k = os.environ.get("GROQ_API_KEY", "")
# print("GROQ_API_KEY exists:", bool(k))
# print("repr:", repr(k))          # 会显示是否带引号/空格/换行
# print("len:", len(k))
# print("strip repr:", repr(k.strip()))
# print("startswith gsk_:", k.strip().startswith("gsk_"))
# print("has leading/trailing whitespace:", k != k.strip())

# # 检查是否有不可见控制字符（换行、制表符等）
# bad = [c for c in k if ord(c) < 32]
# print("control chars:", [repr(c) for c in bad])

import os
from openai import OpenAI

api_key = os.environ.get("GROQ_API_KEY", "").strip()
# api_key = api_key.replace("gsk_gsk_", "gsk_")  # 防呆

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

resp = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Say ping and nothing else."}],
    temperature=0.2,
)

print(resp.choices[0].message.content)

