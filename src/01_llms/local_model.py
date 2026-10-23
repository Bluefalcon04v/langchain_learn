from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
# 
# Before Running the local models start the ollama on your terminal 
# by -> checking if the ollama you're going to use is already install ie "ollama pull llama3.2"
# -> and starting it by -> "ollama serve"
# 

    
from langchain_ollama import ChatOllama
from utils.helper import print_title, print_separator

def main()-> None:
    print_title('Local Ollama Model')
    
    llm = ChatOllama(
        model='llama3.2',
        temperature=0.7
    )
    
    prompt="""
        Explain me the using the local ollama model compare to the online models
    """
    
    print_title("USER PROMPT")
    print(prompt.strip())
    
    print("\nGenerating Response\n")
    response = llm.invoke(prompt)
    
    print_separator()
    print(response.content)
    
if __name__ == "__main__":
    main()
    
    
