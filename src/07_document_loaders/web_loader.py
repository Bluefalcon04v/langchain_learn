from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from utils.helper import print_separator, print_title
from langchain_community.document_loaders import WebBaseLoader

def main():
    print_title("Web Loader")
    
    loader = WebBaseLoader(web_path=("https://www.flipkart.com/arihant-rajvyavastha-vigyan-iithas-bhoogol-arthvyastha-static-gk-capsule-handwritten-notes-exam-2026-27-ssc-railway-police-nda-cds-capf-state-pcs-other-exams-short-smart-preparation-hindi-medium/p/itm9297886927914?pid=9789368403517&lid=LSTBOK9789368403517RAEFE1&marketplace=FLIPKART&q=arihant&store=bks&srno=s_1_1&otracker=search&otracker1=search&fm=organic&iid=en_J2_qfyRn4Bs32fu2te0eIJt3-I-havMWdcd0YaSQuRz0EQFtIEJJvQ3te9nciezo-XVngSwo_Vl4iiK4JL5r8rcK2yC4JUunwPDV9vy3X3Ylz8nnG3SyDz5d6HTnKdXt&ppt=hp&ppn=homepage&ssid=lmsevvlts00000001793267259651&qH=22a7824e4e2c170f&ov_redirect=true&ov_redirect=true"))
    documents = loader.load()
    
    print(f"Total Documents Loaded: {len(documents)}")
    print_separator()
    
    document = documents[0]
    print(f"Document Metadata: {document.metadata}")
    print_separator()
    
    print(f"Document Content :\n  {document.page_content[:1000]}")    
    
if __name__ == "__main__":
    main()