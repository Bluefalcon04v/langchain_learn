from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_classic.memory import ConversationSummaryMemory
from utils.helper import print_separator, print_title
from langchain_ollama import ChatOllama
from llm_client import get_llm

def main():
    print_title("Conversation Summary Memory")
    llm = ChatOllama(model="llama3.2")
    # llm = get_llm()
    
    memory = ConversationSummaryMemory(
        llm=llm
    )
    
    conversations=[
        ("Hi!", "Hello"),
        ("My name is Anuj", "Hi Anuj i'm AI"),
        ("I work as a Developer", "Good to know"),
        ("Im a google dev", "It is a very good MNC"),
        ("Im learning AI Engineering", "Good to know that"),
    ]
    
    for human, ai in conversations:
        memory.save_context(
            {"input":human},
            {"output":ai}
        )
    history = memory.load_memory_variables({})
    
    print(history["history"])
    print_separator()
        
if __name__ == "__main__":
    main()    
    