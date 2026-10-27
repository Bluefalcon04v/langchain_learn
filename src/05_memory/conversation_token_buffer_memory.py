
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_classic.memory import ConversationTokenBufferMemory
from utils.helper import print_separator, print_title
from langchain_ollama import ChatOllama
from llm_client import get_llm

def main():
    print_title("Conversation Token Buffer Memory")

    llm = ChatOllama(model="llama3.2")
    # llm = get_llm()

    memory = ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=50,
        return_messages=True,
    )

    conversations = [
        ("Hi!", "Hello"),
        ("My name is Anuj", "Hi Anuj, I'm an AI."),
        ("I work as a Developer", "Good to know."),
        ("I'm a Google developer", "It is a very good company."),
        ("I'm learning AI Engineering", "Good to know that."),
    ]

    for human, ai in conversations:
        memory.save_context(
            {"input": human},
            {"output": ai},
        )

    history = memory.load_memory_variables({})

    for index, message in enumerate(history["history"], start=1):
        print(f"{message.type.capitalize()}: {message.content}")
        print_separator()


if __name__ == "__main__":
    main()
