import streamlit as st
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer

# in this model if you entered half sentence it will generate the rest of the sentence kind of story telling

# 1. Load the tokenizer explicitly to fix the padding issue
model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id  # GPT-2 doesn't have a pad token by default

# 2. Load the pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 256,
        "pad_token_id": tokenizer.pad_token_id, # Essential for GPT-2
        "do_sample": True # Temperature requires do_sample to be True
    },
    # Pass the tokenizer we configured
    model_kwargs={
        "device_map": "auto",
        "torch_dtype": "auto",
    }
)

st.header('GPT-2 Text Generation')
user_input = st.text_area('Enter your prompt here:')

if st.button('Generate'):
    if user_input.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            # 3. Use invoke() instead of run()
            output = llm.invoke(user_input)
            st.write(output)