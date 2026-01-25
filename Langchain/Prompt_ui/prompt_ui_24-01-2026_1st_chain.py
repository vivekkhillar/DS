# Build a chain to avoid multiple invoke method

from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace,HuggingFacePipeline
from langchain_core.prompts import load_prompt,PromptTemplate
import streamlit as st

# create LLM:

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    model_kwargs = {
        "device_map" : "auto", # Auto place layers on CPU/GPU
        "torch_dtype" : "auto" # Auto pick float16 if GPU, else float32
    },
    pipeline_kwargs={
        "temperature" : 1.7
    }

)

model = ChatHuggingFace(llm =llm)

st.header('Research tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )
style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 
length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt("prompt_generalize.json")

if st.button('Summarize'):
    # chain created for to load the model and template then invoke the loaded templated to the model and capture the response
    # this chain creation also need to be done in a correct direction 1st load template then invoke model
    chain = template | model
    response = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write("Summary:")
    
    st.write(response.content.split("<|assistant|>")[-1])