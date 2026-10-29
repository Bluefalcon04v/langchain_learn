from pathlib import Path 
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helper import print_separator, print_title
from langchain_text_splitters import MarkdownHeaderTextSplitter

def main():
    print_title("Markdown Splitter")    
    file_path = PROJECT_ROOT / "data" / "input" / "langchain_notes.md"
    
    with open(file_path, "r", encoding="utf-8") as file:
        markdown_text = file.read()
        
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )
    
    document = splitter.split_text(markdown_text)    
    print(f"Chunks Created: {len(document)}")
    print_separator()
    
    for index, doc in enumerate(document, start=1):
        print(f"Chunk -> {index}")
        print(f"\nMetadata: \n {doc.metadata}")
        print(f"\nContent: \n {doc.page_content}")
        print_separator()
    
if __name__ == "__main__":
    main()