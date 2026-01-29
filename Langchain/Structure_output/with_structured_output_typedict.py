# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# from typing import TypedDict
# import os
# load_dotenv()

# HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# llm = HuggingFaceEndpoint(
#     repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
#     task = "text-generation",
#     huggingfacehub_api_token=HF_TOKEN,
#     max_new_tokens=256,
#     temperature=0.1
# )

# model = ChatHuggingFace(llm = llm)

# class review(TypedDict):
#     sentiment: str


# structured_output = model.with_structured_output(review)
# print (structured_output)
# try:

#     result = structured_output.invoke("This quantum computing tutorial was absolutely brilliant! "
#     "It explained complex concepts in a simple way. I'd give it 5 stars. "
#     "Very helpful for beginners.")
#     print (result)
# except Exception as e:
#     print(f"Error occurred: {e}")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="bastienp/Gemma-2-2B-Instruct-structured-output",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=256,
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    title: str
    rating: int
    sentiment: str

structured_llm = model.with_structured_output(Review)

result = structured_llm.invoke(
    "Extract: This quantum computing tutorial was brilliant! 5 stars."
)
print(result)