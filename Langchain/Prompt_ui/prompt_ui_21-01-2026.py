import streamlit as st
from openai import OpenAI
import httpx

class Config:
    LLM_API_URL = "https://10.173.119.32/v1"   # ✅ FIXED
    LLM_MODEL_NAME = "gemma3:12b"
    MAX_TOKENS = 1024
    TEMPERATURE = 0.1
    TIMEOUT = 180.0

@st.cache_resource
def get_client():
    http_client = httpx.Client(
        verify=False,           # only if self-signed cert
        timeout=Config.TIMEOUT
    )
    return OpenAI(
        base_url=Config.LLM_API_URL,
        api_key="dummy",
        http_client=http_client
    )

client = get_client()

st.title("Research Tool")

user_input = st.text_area("Enter your query", height=200)

if st.button("Summarize"):
    if not user_input.strip():
        st.warning("Please enter a query")
    else:
        messages = [{"role": "user", "content": user_input}]
        output = st.empty()
        text = ""

        try:
            with client.chat.completions.stream(
                model=Config.LLM_MODEL_NAME,
                messages=messages,
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
            ) as stream:
                for event in stream:
                    if event.type == "content.delta":
                        text += event.delta
                        output.markdown(text)

        except Exception as e:
            st.error(str(e))
