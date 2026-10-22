from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm
from utils.helper import print_title, print_separator

def chat_with_model() -> None:
    """
    Sends a simple prompt to the configured chat model and prints the response
    """
    
    llm = get_llm()
    
    prompt = """
    Explain me the Langchain in 2 concise bullet points
    """
    print_title("USER PROMPT")
    print(prompt.strip())
    
    print("\nGenerating Response\n")
    response = llm.invoke(prompt)
    
    print_separator() 
    print("AI Response\n")
    print(response.content)
    print_separator() 

if __name__ == "__main__":
    chat_with_model()