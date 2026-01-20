from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation",
                                        pipeline_kwargs={"temperature": 0.7, "max_new_tokens": 256})
model = ChatHuggingFace(llm=llm)

results = model.invoke("who is the prime minister of india.")
print(results.content)