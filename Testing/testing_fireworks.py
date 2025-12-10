# test_fireworks.py
from openai import OpenAI

'''client = OpenAI(
    api_key="fw_3gsd",  # Your full key
    base_url="https://api.fireworks.ai/inference/v1"
)'''

import os
from openai import OpenAI

client = OpenAI(
    api_key= "fw_3ZJyed",
    base_url="https://api.fireworks.ai/inference/v1"
)

try:
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        messages=[{"role": "user", "content": "Say hello!"}],
        stream=False
    )
    print("✅ Non-streaming works!")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("❌ Error:", e)

# Test streaming (like your format)
print("\n🔥 Testing streaming:")
stream = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    messages=[{"role": "user", "content": "Tell me a short joke"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print("\n✅ Streaming works!")