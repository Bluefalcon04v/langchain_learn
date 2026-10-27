from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_classic.memory import ConversationBufferMemory
from utils.helper import print_separator, print_title

def main():
    print_title("Conversation Buffer Memory")
    
    memory = ConversationBufferMemory(return_messages=True)
    
    memory.save_context(
        {"input": "Hi My name is Anuj"},
        {"output": "Hi Anuj i'm AI"}
    )
    
    memory.save_context(
        {"input": "I work as a Developer"},
        {"output": "Good to know"}
    )
    
    memory.save_context(
        {"input": "Im a google dev"},
        {"output": "It is a very good MNC"}
    )
    
    print("\n Conversation History\n")
    history = memory.load_memory_variables({})
    for input, message in enumerate(history["history"], start=1):
        print(f"{message.type.capitalize()}: {message.content}")
        print_separator()
        
if __name__ == "__main__":
    main()