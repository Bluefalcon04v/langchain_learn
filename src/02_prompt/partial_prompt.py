from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.prompts import PromptTemplate
from utils.helper import print_separator, print_title
from llm_client import get_llm

def main():
    print_title("Partial Prompt")
    llm = get_llm()
    
    prompt_template = PromptTemplate(
       template="""
You are an expert {domain} trainer.

Explain the concept of the {topic}

Keep the explanation under {word_limit} words.        
        """,
        input_variables = ["domain", "topic", "word_limit"]        
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
    
    print(prompt.text)
    
    print_separator()
    
    response = llm.invoke(prompt)
    
    print("\n LLM Response\n")
    print(response.content)
    
if __name__ == "__main__":
    main()