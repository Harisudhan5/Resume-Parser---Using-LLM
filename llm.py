from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

p = llm.invoke("Get the Name, Phone no, github, linkedin, from the following content and return it as json format")

print(p)
