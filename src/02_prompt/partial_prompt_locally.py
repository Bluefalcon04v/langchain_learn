from langchain_ollama import ChatOllama
from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.prompts import ChatPromptTemplate
from utils.helper import print_separator, print_title

def main():
    print_title("Partial Prompt Locally")
    llm = ChatOllama(
        model="llama3.2"
    )
    
    prompt_template = ChatPromptTemplate.from_template(
        """
You are an expert {domain} trainer.

Explain the concept of the {topic}

Keep the explanation under {word_limit} words.        
        """,
    )
    
    partial_prompt = prompt_template.partial(
        domain="Generative AI"
    )    
    
    prompt = partial_prompt.invoke(
        {
            "topic": "Vector Embeddings",
            "word_limit": 200
        }
    )
    
    print(prompt.to_string())
    
    print_separator()
    
    response = llm.invoke(prompt)
    
    print("\n LLM Response\n")
    print(response.content)
    
if __name__ == "__main__":
    main()