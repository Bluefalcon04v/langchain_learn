from langchain_ollama import ChatOllama
from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.prompts import ChatPromptTemplate
from utils.helper import print_separator, print_title

def main() -> None:
    print_title("Few Shot Prompt")
    
    llm = ChatOllama(
        model="llama3.2"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an AI Assistance that categorizes customers support tickets.
Classify each ticket into exactly one category form the following:
- Billing
- Technical Issue
- Account Management
- Feature Request
- General Inquiry

Respond Only with the category name.
"""
            ),        
            (
                "human",
"""
Example1:

Ticket:
I was charged twice for my monthly subscription 
category: 
Billing


Example2:

Ticket:
I forgot my password and cant log into my account  
category: 
Account Manager


Example3:

Ticket:

The mobile app crashed every time I upload a photo: 
Technical Issue. 

Now classify the following review:
{ticket}

"""
            ),
            (
                "ai",
                "Sure! Here is the result."
            )
        ]
    )
    
    messages = prompt.invoke(
        {   
            "ticket":(
                "It would be great if your platform could support"
                "dark mode for the dashboard."
            )
         }
    )
    
    for message in messages.messages:
        print(f"{message.type.upper()}")
        print(message.content)
        print()
        
    print_separator()
    
    response = llm.invoke(messages)
    
    print("LLMs RESPONSE\n")
    print(response.content)
    
if __name__ == "__main__":
    main()