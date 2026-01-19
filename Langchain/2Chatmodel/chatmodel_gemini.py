from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()   

chat_llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.7)

result = chat_llm.invoke("Hello, how are you?")

print(result)
