from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv() 
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Simple dynamic messgae creation to send in the model
# chat_template = ChatPromptTemplate.from_messages([
#     # SystemMessage(content="You are a helpful domain {domain} expert"),
#     # HumanMessage(content="Explain in simpel terms, what is {topic}?")
#     ('system' , 'You are a helpful domain {domain} expert'),
#     ('human' , 'Explain in simpel terms, what is {topic}?')
# ])

# prompt = chat_template.invoke({'domain': "science", "topic": "quantum computing"})

# print (prompt)


# to Dynamic Message Creation have some steps

# 1. Chat template

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])


chat_history = []
# 2. load chat_history
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

# 3. Create LLM
llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    task = 'text-generation',
    huggingfacehub_api_token=HF_TOKEN
)

model = ChatHuggingFace(llm=llm)

# 3. Create Prompt

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    prompt = chat_template.invoke({
        'chat_history': chat_history,
        'query': user_input
        })

    response = model.invoke(prompt)
    print(f"Assistant: {response.content}")
