from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chat_llm = ChatAnthropic(model='claude-3-5-sonnet-20241022', temperature=0.7)

result = chat_llm.invoke("Hello, how are you?")

print(result)
