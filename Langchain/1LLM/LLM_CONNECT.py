from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

LLM = OpenAI(model ='gpt-4')

result = LLM.invoke("Hello, how are you?")

print(result)