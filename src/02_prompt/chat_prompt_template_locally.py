from langchain_ollama import ChatOllama
from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.prompts import ChatPromptTemplate
from utils.helper import print_separator, print_title

def main()-> None:
    print_title("Chat Prompt Template")
    
    # i have implemented it using local model use the llm = get_llm()
    # to invoke the openAI API key 
    
    llm = ChatOllama(
        model="llama3.2"
    )
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI experienced in {profession}. Your name is {name}."
            ),
            (
                "human",
                "Explain the concept of {topic} in less than {word_limit} words"
            ),
            (
                "ai",
                "Sure! Here is the result."
            ),
        ]
    )
    
    messages = chat_prompt.invoke(
        {
            "name": "jarvis",
            "profession": "AI Engineer",
            "topic": "Explain me about AI, what is your name ?",
            "word_limit": 100
        }
    )
    
    print("Formatted Message\n")
    
    for message in messages.messages:
        print(f"{message.type.upper()}")
        print(message.content)
        print_separator()
        
    response = llm.invoke(messages)
    print(response.content)
        

if __name__ == "__main__":
    main()