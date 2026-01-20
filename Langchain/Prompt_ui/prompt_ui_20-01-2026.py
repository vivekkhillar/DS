from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st

# Load the model safely
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={
        "device_map": "auto",       # Auto place layers on CPU/GPU
        "torch_dtype": "auto",      # Auto pick float16 if GPU, else float32
        "load_in_8bit": True        # Optional: reduces memory usage for smaller GPUs
    },
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 256
    }
)

# Wrap for chat output
model = ChatHuggingFace(llm=llm)

st.header('Research tool')
user_input = st.text_area('Enter your query here')

if st.button('Summarize'):
    if user_input.strip() == "":
        st.warning("Please enter a query first.")
    else:
        result = model.invoke(user_input)
        print(result.content)
        st.write(result.content)
