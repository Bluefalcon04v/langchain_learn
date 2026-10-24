from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from llm_client import get_llm
from utils.helper import print_separator, print_title

def main()->None:
    print_title("Messages Placeholder")
    
    llm = get_llm()
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI tutor who answers questions based on the ongoing conversation."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )
    
    chat_history = [
        HumanMessage(content="My name is Bruce"),
        AIMessage(
            content=(
                "Hello,  Bruce how can i help you?"
            )
        ),
        HumanMessage(content="Im a Frontend developer"),
        AIMessage(
            content=(
                "It's great!"
            )
        )        
    ]    
    
    messages = chat_prompt.invoke(
        {
            "chat_history": chat_history,
            "question": "What is my name and what do i do?"
        }
    )
    
    response = llm.invoke(messages)
    
    print("\n LLM Response\n")
    print(response.content)
    
if __name__ =="__main__":
    main()