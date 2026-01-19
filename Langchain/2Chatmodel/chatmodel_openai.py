from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

# the temperature setting controls the randomness of the output and more creative 
# if within 0.0 -0.3 then Factual answers
# if within 0.4 - 0.7 then Balanced answers
# if within 0.8 - 1.0 then Creative answers, story telling
# if 1.5+ - then maximum randomness

chat_llm = ChatOpenAI(model='gpt-4', temperature=0.7)

result = chat_llm.invoke("Hello, how are you?")

print(result)