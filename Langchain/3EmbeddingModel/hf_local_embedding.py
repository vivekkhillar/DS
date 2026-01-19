from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = ["Delhi is the capital of INDIA","Kolkata is the capital of West bengal", "BBSR is the capital of Odisha"]
vector = embeddings.embed_documents(text)
print(str(vector))