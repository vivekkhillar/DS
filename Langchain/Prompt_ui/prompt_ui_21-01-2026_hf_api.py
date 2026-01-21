from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

@st.cache_resource
def get_model():
    return HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        huggingfacehub_api_token=HF_API_KEY,
        task="text-generation",
        provider="hf-inference",   # ⭐ THIS FIXES IT
        max_new_tokens=256,
        temperature=0.7
    )


model = get_model()

st.header("Research Tool")
user_input = st.text_area("Enter your query here")

if st.button("Summarize"):
    if not user_input.strip():
        st.warning("Please enter a query first.")
    else:
        result = model.invoke(user_input)
        st.write(result)
