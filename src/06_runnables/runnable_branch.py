from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch

from llm_client import get_llm
from langchain_ollama import ChatOllama

from utils.helper import print_separator, print_title

def main():
    print_title("Demonstrates Runnable Branch")
    
    llm = ChatOllama(model="llama3.2")
    
    technical_chain = (
        PromptTemplate.from_template(
            "Answer the following technical question\n\n {query}"
        ) 
        | llm
        | StrOutputParser()
    )
    
    general_chain = (
        PromptTemplate.from_template(
            "Answer the following general question\n\n {query}"
        ) 
        | llm
        | StrOutputParser()
    )
    
    branch = RunnableBranch(
        (
            lambda x: any(
                keyword in x["query"].lower()
                for keyword in [
                    "python", "langchain", "Ai", "llm", "rag"
                ]
            ), 
            technical_chain
        ),
         general_chain
    )
    
    query = "what is retrieval augmented generation"
    
    print(f"Query:\n {query}")
    print_separator()
    
    response = branch.invoke({"query": query})
    
    print("Response \n")
    print(response)
    
if __name__== "__main__":
    main()