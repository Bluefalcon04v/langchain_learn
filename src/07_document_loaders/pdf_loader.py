from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_community.document_loaders import PyPDFLoader
from utils.helper import print_separator, print_title

def main():
    print_title("PDF Loader")
    file_path = PROJECT_ROOT / "data" / "input" / "sample.pdf"
    
    loader = PyPDFLoader(file_path=file_path)
    
    document =  loader.load()
    print(f"Total PDF Loaded: {len(document)}")
    print_separator()
    
    first_page = document[1]
    print(f"Page Metadata: {first_page}")
    print_separator()
    
    print(f"First Page Content: {first_page.page_content}")
    
if __name__ == "__main__":
    main()
    