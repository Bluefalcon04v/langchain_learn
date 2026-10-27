from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_classic.memory import ConversationBufferWindowMemory
from utils.helper import print_separator, print_title


def main():
    print_title("Conversation Buffer Window Memory")
    memory = ConversationBufferWindowMemory(k=2, return_messages=True)
    
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
    for input, message in enumerate(history["history"], start=1):
        print(f"{message.type.capitalize()}: {message.content}")
        print_separator()
        
if __name__ == "__main__":
    main()