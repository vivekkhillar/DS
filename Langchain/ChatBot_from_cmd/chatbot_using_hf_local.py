from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


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

while True:

    user_input = input("you: ")
    if user_input.lower() == "exit":
        break
    response = model.invoke(user_input)
    print(f"Model: {response}")