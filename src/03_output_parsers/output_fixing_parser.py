from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser

from llm_client import get_llm
from langchain_ollama import ChatOllama
from utils.helper import print_separator, print_title

class Employee(BaseModel):
    name: str = Field(description="Employee's full name")
    department :str = Field(description="Department name")
    experience :int = Field(description="Years of experience")
    skills:list[str] = Field(description="List of technical skills")
    
def main():
    print_title("Output Fixing Parser")
    
    llm = ChatOllama(model="llama3.2")
    parser = PydanticOutputParser(pydantic_object=Employee)
    
    fixing_parser = OutputFixingParser.from_llm(
        parser=parser,
        llm=llm
    )
    
    malformed_output = """
    {
        "name": "Bruce Sam",
        "department": "AI",
        "experience": "5 years",
        "skills":"frontend, react, js, python, ml-ai"
    }
    """
    print("Malformed Output:\n")
    print(malformed_output)
    
    print_separator()
    
    parsed_response= fixing_parser.invoke(malformed_output)
    
    print("Fixed Parsed Response\n")
    print(parsed_response)
    
    print_separator()
    
    print(f"Name: {parsed_response.name}")
    print(f"Department : {parsed_response.department}")
    print(f"Experience : {parsed_response.experience}")
    print(f"Skills : {parsed_response.skills}")
        
    
if __name__== "__main__":
    main()