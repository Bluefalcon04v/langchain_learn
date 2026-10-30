from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helper import print_separator, print_title

def main():
    print_title("Vector Store Retrievers")
    loader = TextLoader(PROJECT_ROOT / "data" / "input" / "sample.txt")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50 
    )

    chunks = splitter.split_documents(documents=documents)    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    retriever = vector_store.as_retriever(search_kwargs = {"k" : 2})
    
    query = "What is my Name?"
    print(f"Query: \n {query}")
    print_separator()
    
    documents = retriever.invoke(query)
    
    for idx, doc in enumerate(documents, start=1):
        print(f"Retriever Documents: {idx} \n")
        print(doc.page_content)
        print_separator()
    
        
if __name__ == "__main__":
    main()
