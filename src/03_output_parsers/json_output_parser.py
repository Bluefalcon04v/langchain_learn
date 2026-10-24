from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from llm_client import get_llm
from langchain_ollama import ChatOllama
from utils.helper import print_separator, print_title


def main():
    print_title("JSON OUTPUT PARSER")
    llm = ChatOllama(model="llama3.2")
    # llm = get_llm()
    
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="""
Extract the following product information.
- Product Name 
- Screen Size
- Screen Type
- Storage
- RAM
- Price
- Chip Specification

{format_instructions}

Product Description:
{product}
        """,
        input_variables=["product"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )
    
    prompt_value = prompt.invoke(
        {
            "product":(
                "The Apple MacBook Air M4 with 16GB"
                "512GB SSD storage,"
                "OLCD Screen, 15inch screen size"
                "price $1599"
            )
        }
    )
    
    print(prompt_value.text)
    print_separator()
    
    response = llm.invoke(prompt_value)
    print(response.content)
    print_separator()
    
    parsed_response = parser.invoke(response)    
    print(parsed_response)
    
    
if __name__ == "__main__":
    main()