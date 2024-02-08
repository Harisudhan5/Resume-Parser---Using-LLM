# Testing llama2 on OLLama to chat with LLM
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")
prompt = ""
result = llm.invoke("Get the Name, Phone no, github, linkedin, from the following content and return it as json format")
print(result)
