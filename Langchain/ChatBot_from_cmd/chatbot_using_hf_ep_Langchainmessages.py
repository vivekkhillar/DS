from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from datetime import datetime
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

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ["exit", "quit"]:

        with open("chat_history.txt", "w", encoding="utf-8") as f:
            for message in chat_history:
                timestamp = datetime.now().strftime("%Y-%m-%d")
                if isinstance(message, SystemMessage):
                    role = "System"
                elif isinstance(message, HumanMessage):
                    role = "Human"
                elif isinstance(message, AIMessage):
                    role = "AI"
                f.write(f"{timestamp} - {role}: {message.content}\n")
        print("Exiting the chat.")
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"Model: {result.content}")