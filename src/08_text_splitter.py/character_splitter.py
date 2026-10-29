from pathlib import Path 
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helper import print_separator, print_title
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

def main():
    print_title("Character Splitter")    
    file_path = PROJECT_ROOT / "data" / "input" / "sample.txt"    
    
    loader = TextLoader(file_path=file_path)
    document = loader.load()
    
    print(f"Original Documents: {len(document)}")
    print_separator()
    
    splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size = 100,
        chunk_overlap = 50
    )
    
    chunks = splitter.split_documents(documents=document)
    print(f"Chunks Created: {len(chunks)}")
    print_separator()
    
    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk -> {index}")
        print(f"Characters: {len(chunk.page_content)}")
        print()
        print(chunk.page_content)
        print_separator()
    
    
if __name__ == "__main__":
    main()