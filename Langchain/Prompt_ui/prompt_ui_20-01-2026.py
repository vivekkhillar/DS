from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

# select the model 

llm = HuggingFacePipeline.from_model_id(model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',task= 'text-generation',
                                        pipeline_kwargs={'temperature': 0.5})

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India?")
print(result.content)