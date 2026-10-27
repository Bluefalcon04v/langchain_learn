from pathlib import Path
import sys 

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

from langchain_ollama import ChatOllama
from utils.helper import print_separator, print_title

def main():
    print_title("Router Chain")
    llm = ChatOllama(model="llama3.2")
    
    science_chain = (PromptTemplate.from_template("Explain the following science concept:\n {query}" )| llm | StrOutputParser())

    coding_chain = (PromptTemplate.from_template("Explain the following coding concept:\n {query}") | llm | StrOutputParser())

    general_chain = (PromptTemplate.from_template("Answer the following general:\n {query}") | llm | StrOutputParser())
    
    def router(info):
        query =  info["query"].lower()
        
        if "python" in query or "code" in query:
            print("Routing to:Coding Chain\n")
            return coding_chain
            
        elif "physics" in query or "science" in query:
            print("Routing it to: Science Chain\n")
            return science_chain
        
        else: 
            return general_chain
        
    chain = RunnableLambda(router)
    
    query = "Give me 5 good practices for code"
    print(f"Query: {query}\n")
    
    response = chain.invoke({"query": query})
    print(f"Response: {response}")
    
if __name__ == "__main__":
    main()