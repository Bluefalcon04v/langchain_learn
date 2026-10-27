from pathlib import Path
import sys 

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

from langchain_ollama import ChatOllama
from utils.helper import print_separator, print_title

def main():
    print_title("Custom Chain")
    llm = ChatOllama(model="llama3.2")
    
    title_chain = (
        PromptTemplate.from_template(
            "Generate a catchy blog title about {topic}."
        )
        | llm
        | StrOutputParser()
    )

    outline_chain = (
        PromptTemplate.from_template(
            """Create a detailed blog outline for the following title.
            
            Title:
            {title}
            """
        )
        | llm
        | StrOutputParser()
    )

    linkedin_chain = (
        PromptTemplate.from_template(
            """
            Write a professional LinkedIn post to promote the following blog.
            
            Title:
            {title}
            
            Outline:
            {outline}
            """
        )
        | llm
        | StrOutputParser()
    )
    
    def blog_generation_workflow(topic:str)-> dict:
        """Executes a multi-step content generation workflow."""
        
        title = title_chain.invoke({"topic": topic})
        
        outline = outline_chain.invoke({"title": title})
        
        linkedin_post = linkedin_chain.invoke(
            {
                "title": title,
                "outline": outline
            }
        )
        
        return {
            "title": title,
            "outline": outline,
            "linkedin_post": linkedin_post
        }
    
    custom_chain = RunnableLambda(blog_generation_workflow)
    
    topic = "AI Agents"
    print(f"Topic: {topic}")
    
    print_separator()
    
    result = custom_chain.invoke(topic)
    
    print("Generated Title\n ")
    print(result["title"])
    print_separator()
    
    print("Generated Outline\n ")
    print(result["outline"])
    print_separator()
    
    print("LinkedIn Post\n ")
    print(result["linkedin_post"])
    print_separator()
    
if __name__ == "__main__":
    main()