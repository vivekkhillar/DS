from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',task = 'text-generation')
chat_hugging_face = ChatHuggingFace(llm=llm)

result = chat_hugging_face.invoke("What is the capital of india?")
print(result.content)