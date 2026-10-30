from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from utils.helper import print_separator, print_title

load_dotenv()

def main():
    print_title("Text Embeddings")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    text = "Langchain Makes it easy to build applications powered by LLMs."
    vector = embeddings.embed_query(text=text)
    
    print(f"Input text: {text}")
    print_separator()
    
    print(f"Embeddings Dimension: {len(vector)}")
    print_separator()
    
    print(f"First 10 vector values: {vector}")    
    
    
if __name__ == "__main__":
    main()