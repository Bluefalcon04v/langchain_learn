from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

from utils.helper import print_separator, print_title

def main():
    print_title("Parent Document Retrievers")
    loader = PyPDFLoader(PROJECT_ROOT / "data" / "input" / "sample.pdf")
    documents = loader.load()
    
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100 
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50 
    )

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )    
    
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    retriever.add_documents(documents=documents)
    
    query = "Explain the Rag Pipeline"
    print(f"Query: {query}")
    
    print_separator()
    
    results = retriever.invoke(query)
    print(f"Retrieved Parent Documents: {len(results)}")
    print_separator()
    
    for idx, doc in enumerate(results, start=1):
        print(f"Parent Document {idx}")
        print(doc.page_content)
        print_separator()
    
if __name__ == "__main__":
    main()
