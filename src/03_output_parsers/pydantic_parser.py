from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from llm_client import get_llm
from utils.helper import print_separator, print_title

class Employee(BaseModel):
    name: str = Field(description="Employee's full name")
    department :str = Field(description="Department name")
    experience :int = Field(description="Years of experience")
    skills:list[str] = Field(description="List of technical skills")
    
def main():
    print_title("Pydantic Output Parsing")
    
    llm = get_llm()
    # llm = ChatOllama(
    #     model="llama3.2"
    # )
    parser = PydanticOutputParser(pydantic_object= Employee)
    
    prompt = PromptTemplate(
        template="""
        Extract the employee information
        
        {format_instructions}
        
        Text: 
        {employee_details}
        """,
        input_variables=["employee_detail"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    prompt_value=prompt.invoke(
        {
            "employee_details":(
                "Bruce works as a AI Engineer in the AI team "
                "He is 10 years of experience in python"
                "Frontend, SQL and Node"
            )
        }
    )
    
    print("Formatted Response")
    print(prompt_value.text)
    
    print_separator()
    
    response = llm.invoke(prompt_value)
    print(response.content)
    
    print_separator()
    
    parsed_response = parser.invoke(response)    
    print(parsed_response)
    
    print_separator()
    
    print(f"Name: {parsed_response.name}")
    print(f"Department : {parsed_response.department}")
    print(f"Experience : {parsed_response.experience}")
    print(f"Skills : {parsed_response.skills}")
    
    
if __name__== "__main__":
    main()