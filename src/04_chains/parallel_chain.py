from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from llm_client import get_llm
from langchain_ollama import ChatOllama
from utils.helper import print_separator, print_title
from langchain_core.runnables import RunnableParallel

def main():
    print_title("Parallel Chain")
    
    llm = ChatOllama(model="llama3.2")
    
    summary_prompt = PromptTemplate.from_template(
        "Write a short summary about {topic}"
    )

    advantages_prompt = PromptTemplate.from_template(
        "List 5 advantages of {topic}"
    )

    applications_prompt = PromptTemplate.from_template(
        "List three real-world applications of {topic}"
    )
    
    parallel_chain = RunnableParallel(
        summary = summary_prompt | llm | StrOutputParser(),
        advantages = advantages_prompt | llm | StrOutputParser(),
        applications = applications_prompt | llm | StrOutputParser(),
    )
    
    topic = "FrontEnd with Reactjs "
    
    print(f"Topic:\n {topic}")
    print_separator()
    
    result = parallel_chain.invoke({"topic": topic})
    
    print("Summary:\n")
    print(result["summary"])
    print_separator()
    
    print("Advantages:\n")
    print(result["advantages"])
    print_separator()

    print("Applications:\n")
    print(result["applications"])
    print_separator()
    
if __name__ == "__main__":
    main()