from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict
import os
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id = "h2oai/h2ogpt-gpt-jing-chat-3.5b",
    task = "text-generation",
    huggingfacehub_api_token=HF_TOKEN
)

model = ChatHuggingFace(llm = llm)

class review(TypedDict):
    title: str
    rating: int 
    review_text: str


structured_output = model.with_structured_output(review)


result = structured_output.invoke(f"Quantum computing is a new way of processing information that\'s different from the computers we use today. Here\'s a simplified explanation:\n\n**Classical Computing (Traditional Computers)**\n\nTraditional computers use bits to process information. A bit is either 0 or 1, like a light switch that\'s either on or off. These bits are combined to perform calculations and store data.\n\n**Quantum Computing**\n\nQuantum computers use something called (quantum bits). Qubits are special because they can be:\n\n1. 0\n2. 1\n3. Both 0 and 1 at the same time (this is called a superposition)\n\nImagine a coin that can be both heads and tails simultaneously.\n\nQubits can also be connected in a way that lets them work together to solve problems more efficiently than classical computers.\n\n**How Quantum Computing Works**\n\n1. Qubits are used to represent data and perform calculations.\n2. Qubits can be in multiple states (like both 0 and 1) at the same time, which allows them to explore multiple possibilities simultaneously.\n3. The qubits are manipulated using quantum gates (like special instructions) to perform calculations and operations.\n4. The qubits are measured to get the final answer.\n\n**Benefits of Quantum Computing**\n\n1. **Speed**: Quantum computers can solve certain problems much faster than classical computers.\n2. **Security**: Quantum computers can break certain types of encryption, but they can also create new, unbreakable encryption methods.\n3. **Optimization**: Quantum computers can find the best solution among many possibilities, which is useful for tasks like logistics and finance.\n\n**Real-world Applications**\n\nQuantum computing has the potential to revolutionize industries like:\n\n1. Healthcare: Simulating complex biological systems to develop new treatments.\n2. Finance: Optimizing investment portfolios and detecting fraud.\n3. Materials Science: Designing new materials with unique properties.\n4. Cybersecurity: Creating unbreakable encryption and protecting sensitive data.\n\nIn summary, quantum computing is a powerful new technology that uses qubits to process information in a way that\'s different from traditional computers. It has the potential to solve complex problems faster and more efficiently, which can lead to breakthroughs in many fields.")
print (result)