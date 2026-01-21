from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    temperature=0.7,
    max_new_tokens=200,
)
chat_hugging_face = ChatHuggingFace(llm=llm)
result = chat_hugging_face.invoke("What is the capital of India?")
print(result)
