from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

p = llm.invoke("Give me a name of popular movie in tamil")

print(p)