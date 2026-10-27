from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

from llm_client import get_llm
from langchain_ollama import ChatOllama
from utils.helper import print_separator, print_title

def main():
    print_title("Demonstrates Runnable Parallel")
    
    llm = ChatOllama(model="llama3.2")
    
    summary_chain = (
        PromptTemplate.from_template(
            "Write a short summary about {topic}."
        ) 
        | llm
        | StrOutputParser()
    )

    advantages_chain = (
        PromptTemplate.from_template(
            "List three advantages about {topic}."
        ) 
        | llm
        | StrOutputParser()
    )

    applications_chain = (
        PromptTemplate.from_template(
            "List real world example {topic}."
        ) 
        | llm
        | StrOutputParser()
    )
    
    parallel_chain = RunnableParallel(
        summary = summary_chain,
        advantages = advantages_chain,
        applications = applications_chain
    )
    
    topic = "FrontEnd Developer"
    print(f"Topic: \n {topic}")
    print_separator()
    
    result = parallel_chain.invoke({"topic": topic})
    print("Summary\n")
    print(result["summary"])
    
    print("advantages\n")
    print(result["advantages"])

    print("applications\n")
    print(result["applications"])
    
if __name__ == "__main__":
    main()