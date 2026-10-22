from llm_client import get_llm
from utils.helper import print_title, print_separator

def main():
    print_title("Langchain chatbot test")
    llm = get_llm()
    
    print(f"LLM Loaded Success: {llm.__class__.__name__}")
    print_separator()
    print("Type 'exit' to 'quit'. \n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print('\n Bye!!')
            break
        
        response = llm.invoke(user_input)
        print(f"\n AI:{response.content}\n\n")
        
    
if __name__ == "__main__":
    main()