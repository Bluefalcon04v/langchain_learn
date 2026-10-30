from pathlib import Path
import sys
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from utils.helper import print_separator, print_title

load_dotenv()

def main():
    print_title("Chroma Vector Store")
    
    file_path = PROJECT_ROOT / "data" / "input" / "sample.txt"
    loader = TextLoader(file_path=file_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50
    )
    
    chunks = splitter.split_documents(documents=documents)
    print(f"Chunks Created: {len(chunks)}")
    print_separator()
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    chroma_db_path = PROJECT_ROOT / "data" / "chroma_db"
    
    if chroma_db_path.exists():
        shutil.rmtree(chroma_db_path)
    
    print("Adding documents to Chroma")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="langchain_demo",
        persist_directory=str(chroma_db_path)
    )
    
    print(f"Document Count in collection: {vector_store._collection.count()}")
    print_separator()
    
    query = "What is GenAi?"
    
    print(f"Query: {query}")
    print("Running Similarities search")
    
    results = vector_store.similarity_search(query=query, k=2)
    print(f"Retrieved {len(results)} documents\n")
    print_separator()
    
    for idx, doc in enumerate(results, start=1):
        print(f"Results {idx}:\n {doc.page_content}")
        print_separator()
        
if __name__ == "__main__":
    main()