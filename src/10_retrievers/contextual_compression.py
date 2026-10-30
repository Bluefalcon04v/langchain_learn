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
from langchain_ollama import ChatOllama
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from utils.helper import print_separator, print_title

def main():
    print_title("Contextual Compression Retrievers")
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
    
    base_retriever = vector_store.as_retriever()
    compressor = LLMChainExtractor.from_llm(ChatOllama(model="llama3.2"))
    
    retriever =ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor= compressor
    )
    
    query = "What is my Name"
    print(f"Query\n: {query}")
    print_separator()
    
    documents = retriever.invoke(query)
    print(f"Retriever {len(documents)} Compressed Documents\n")
    print_separator()
    
    for idx, doc in enumerate(documents, start=1):
        print(f"Document {idx}\n")
        print(doc.page_content)
        print_separator()
    
if __name__ == "__main__":
    main()
