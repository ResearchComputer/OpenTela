import openai

client = openai.Client(api_key="xxx", base_url="http://148.187.108.172:8092/v1/service/llm/v1")
res = client.chat.completions.create(
    model="swiss-ai/Apertus-70B-Instruct-2509",
    messages=[
        {
            "content": "Who is Alan Turing?", 
            "role": "user",
        }
    ],
    stream=True,
)

for chunk in res:
    if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)