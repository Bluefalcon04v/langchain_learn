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

def main():
    print_title("Demonstrates Runnable Sequences")
    
    llm = ChatOllama(model="llama3.2")
    
    prompt = PromptTemplate.from_template(
        "Explain {topic} in less than 100 words"
    )

    sequence = prompt | llm | StrOutputParser()
    
    topic = "Vector Databases"
    
    print(f"\n Topic :\n {topic}")
    
    print_separator()
    
    response = sequence.invoke({"topic": topic})
    print("Response\n")
    print(response)
    
if __name__ == "__main__":
    main()