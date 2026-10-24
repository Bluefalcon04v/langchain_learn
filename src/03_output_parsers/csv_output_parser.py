from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

from llm_client import get_llm
from langchain_ollama import ChatOllama
from utils.helper import print_separator, print_title

def main():
    print_title("CSV Output Parser")
    
    llm = ChatOllama(model="llama3.2")
    parser = CommaSeparatedListOutputParser()
    
    prompt = PromptTemplate(
        template="""
List the top 10 skills needed to become a AI Engineer 

{format_instructions}
        """,
        input_variables=[],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )
    
    prompt_value = prompt.invoke({})
    
    print("\n Formatted Prompt:\n")
    print(prompt_value.text)
    
    print_separator()
    
    response=llm.invoke(prompt_value)
    
    print("Raw LLM Response ")
    print(response.content)
    
    print_separator()
    
    skills = parser.invoke(response)
    print("\nOutput Parser\n")
    
    for index, skill in enumerate(skills, start=1):
        print(f"{index}, {skill}")
        
        
if __name__ =="__main__":
    main()
    