from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
text = ["Delhi is the capital of INDIA","Kolkata is the capital of West bengal", "BBSR is the capital of Odisha"]
query = 'what is the capital of india'

doc_embedding = embedding.embed_documents(text)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]
index,scores = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]
print(text[index])