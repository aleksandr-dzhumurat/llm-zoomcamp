import requests
import json

url = "http://localhost:11434/api/chat"
query = "What's the formula for energy?"

from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

def llm(prompt, temperature=0.0):
    model = 'gemma:2b'  # 'phi3'
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    print('Num completion_tokens: %s' % response.usage)
    
    return response.choices[0].message.content


# query = "10 * 10"

# print(llm(query))

query = "What's the formula for energy?"
print(llm(query))


# with open('data_gemma.json', 'w') as file:
#     json.dump(resp, file, indent=4)
# payload = {
#     "model": "gemma:2b",
#     "options": {
#         "temperature": 0
#     },
#     "messages": [
#         {
#             "role": "user",
#             "content": query
#         }
#     ],
#     "stream": False
# }

# # Send the POST request
# response = requests.post(url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
# model_response = response.json()
# # Check the response
# if response.status_code == 200:
#     print("Response:", response.text)
# else:
#     print("Failed with status code:", response.status_code)
#     print("Response:", model_response)

# print('Num tokens: %d' % model_response["eval_count"])

# with open('data.json', 'w') as file:
#     json.dump(model_response, file, indent=4)

# payload = {
#     "model": "gemma:2b",
#     "options": {
#         "temperature": 0
#     },
#     "messages": [
#         {
#             "role": "user",
#             "content": query
#         }
#     ],
#     "stream": False
# }
# response = requests.post(url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
# model_response = response.json()
# print('10*10: %s' % model_response["message"]['content'])