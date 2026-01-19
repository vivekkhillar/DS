from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(model_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation",
                                        pipeline_kwargs={"temperature": 0.7, "max_new_tokens": 256})
model = ChatHuggingFace(llm=llm)

results = model.invoke("who is the prime minister of india.")
print(results.content)