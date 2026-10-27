from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from llm_client import get_llm
from langchain_ollama import ChatOllama
from utils.helper import print_separator, print_title

def main():
    print_title("Sequential Chain")
    
    llm = ChatOllama(model="llama3.2")
    
    title_prompt = PromptTemplate.from_template(
        "Generate a catchy blog title about {topic} for GenZ users"
    )
    
    outline_prompt = PromptTemplate.from_template(
        """
        Create a blog outline for the following title
        title: {title}
        Target group is GenZ audience
        
        """
    )   
    
    title_chain = title_prompt | llm | StrOutputParser()
    outline_chain = outline_prompt | llm | StrOutputParser()
    
    topic = "Fat loss Program"
    print(f"Topic:\n {topic}")
    
    print_separator()
    
    title = title_chain.invoke({"topic": topic})
    print(f"Generated title:\n {title}")
    print_separator()
    
    outline = outline_chain.invoke({"title": title})
    print(f"Generated outline:\n {outline}")
    print_separator()
    
if __name__ == "__main__":
    main()