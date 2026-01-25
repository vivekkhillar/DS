from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token=HF_TOKEN, 
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
# collecting chat history in a dictionary and sending whole dictionary to the model but this is not a correct way
chat_history = []
while True:

    user_input = input("You: ")
    chat_history.append({"role": "user", "content": user_input})
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break
    result = model.invoke(chat_history)
    chat_history.append({"role": "assistant", "content": result.content})
    print(f"Model: {result.content}")

print(chat_history)