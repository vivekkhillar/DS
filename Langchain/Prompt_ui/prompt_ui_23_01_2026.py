from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate

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
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = PromptTemplate(
    template="""Please summarize the research paper titled "{paper_input}" with the following
        specifications:
        Explanation Style: {style_input}
            Explanation Length: {length_input}
            1. Mathematical Details:
            - Include relevant mathematical equations if present in the paper.
            - Explain the mathematical concepts using simple, intuitive code snippets
            where applicable.
            2. Analogies:
            - Use relatable analogies to simplify complex ideas.
            If certain information is not available in the paper, respond with: "Insufficient
            information available" instead of guessing.
            Ensure the summary is clear, accurate, and aligned with the provided style and
            length.""",
            input_variables=["paper_input", "style_input", "length_input"]
            )
# fill template variables
prompt = template.invoke(
    {
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    }
)

if st.button('Summarize'):
    st.write("Generating summary...")
    response = model.invoke(prompt)
    st.write("Summary:")
    st.write(response.content)
