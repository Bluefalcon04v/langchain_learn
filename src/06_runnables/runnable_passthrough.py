from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from llm_client import get_llm
from langchain_ollama import ChatOllama

from utils.helper import print_separator, print_title

def main():
    print_title("Demonstrates Runnable Pass Through")
    
    llm = ChatOllama(model="llama3.2")
    
    summary_chain = (
        PromptTemplate.from_template(
            "write a short summary about: {topic}"
        )
        | llm
        | StrOutputParser()
    )
    
    pipeline = RunnableParallel(
        original_input = RunnablePassthrough(),
        summary = summary_chain
    )
    
    topic = "Large Language Models"
    
    print(f"Input Topics:\n {topic}")
    print_separator()
    
    result = pipeline.invoke({"topic": topic})
    
    print("Original Input: \n")
    print(result["original_input"])
    print_separator()
    
    print("Generated Summary: \n")
    print(result["summary"])
        
if __name__ == "__main__":
    main()