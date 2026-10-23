from langchain_ollama import ChatOllama
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import ChatPromptTemplate
from utils.helper import print_separator, print_title 

def main()-> None:
    print_title("ONE SHOT PROMPTING")
    
    llm = ChatOllama(
        model="llama3.2"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert sentiment analysis assistant."
            ),        
            (
                "human",
"""
Classify the sentiment as Positive, Negative, or Neutral

Example:

Review: "The laptop is fast, lightweight, and has a amazing battery life."
Sentiment: Positive 

Now classify the following review:
Review: "{review}"

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
            "review": "The Phone looks premium, its battery charges quickly"
        }
    )
    
    print("FORMATTED PROMPT\n")
    
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