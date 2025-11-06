import requests
import json

# Replace with your actual API key
API_KEY = "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"

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    data=json.dumps({
        "model": "deepseek/deepseek-chat-v3.1:free",  # Optional
        "messages": [
            {
                "role": "user",
                "content": "what is 2+2 "
            }
        ]
    })
)

# Check if the request was successful
if response.status_code == 200:
    print("Success!")
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)