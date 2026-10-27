from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from utils.helper import print_separator, print_title
from langchain_ollama import ChatOllama

def main():
    print_title("Chat Message History")
    chat_history = InMemoryChatMessageHistory()
    
    chat_history.add_message(
        HumanMessage(content="Hi My name is Anuj")
    )

    chat_history.add_message(
        AIMessage(content="Hi Anuj i'm AI")
    )

    chat_history.add_message(
        HumanMessage(content="I work as a Developer")
    )
    
    chat_history.add_message(
        AIMessage(content="Good to know")
    )
    
    chat_history.add_message(
        HumanMessage(content="Im a google dev")
    )
    
    chat_history.add_message(
        AIMessage(content="It is a very good MNC")
    )
    
    print("Conversation History\n")
    
    for index, message in enumerate(chat_history.messages, start=1):
        print(f"{(message.type.capitalize())}: {message.content}")
       
       
if __name__ == "__main__":
    main()