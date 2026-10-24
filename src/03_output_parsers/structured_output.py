from langchain_ollama import ChatOllama
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from llm_client import get_llm
from utils.helper import print_separator, print_title
from langchain_ollama import ChatOllama

# JSON schema 
movie_review_schema={
    "title": "MovieReview",
    "description":  "Schema for a movie review",
    "type": "object",
    "properties":{
        "movie_name": {
            "type": "string",
            "description": "Name of the movie"
        },
        "rating":{
            "type":"number",
            "description":"Rating out of 10"
        },
        "summary":{
            "type":"string",
            "description":"Short review of the movie."
        },
        "genres":{
            "type":"array",
            "items":{"type":"string"},
            "description":"List of movie genres."
        }
    },
    "required":["movie_name", "rating", "summary", "genres"]
}

def main():
    print_title("Description structured output")
    
    # llm = get_llm().with_structured_output(movie_review_schema)
    llm = ChatOllama(
        model="llama3.2"
    ).with_structured_output(movie_review_schema)
    
    response = llm.invoke(
        """
        Review the movie Stree 2
        
        keep it under 50 words and rate it out of 10
        """
    )
    
    print("\nStructure Response\n")
    print(response)
    
    print_separator()
    
    print("\n Movie Data \n")
    print(f"Movie Name: {response['movie_name']}")
    print(f"Rating : {response['rating']}")
    print(f"Genres : {response['genres']}")
    print(f"Summary : {response['summary']}")
        
    
if __name__ == "__main__":
    main()