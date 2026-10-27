from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

from utils.helper import print_separator, print_title

def to_uppercase(text):
    return text.upper()

def count_word(text):
    return f"Total words: {len(text.split())}"

def main():
    print_title("Demonstrates Runnable Parallel")
    
    uppercase = RunnableLambda(to_uppercase)
    word_counter = RunnableLambda(count_word)
    
    text = "Langchain Hello world"
    
    print(f"Original Text\n {text}")
    print_separator()
    
    print(f"Uppercase Text\n")
    print(uppercase.invoke(text))
    print_separator()
    
    print(f"Word Counter\n")
    print(word_counter.invoke(text))
    print_separator()
    
if __name__ == "__main__":
    main()    
    