import json 
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r") as file:
        return json.load(file)
    
def get_llm():
    config = load_config()
    provider = config["provider"].lower()
    
    if provider == "openai":
        return ChatOpenAI(
            model= config["openai"]["model"],
            temperature= config["openai"]["temperature"],
            max_tokens = config["openai"]["max_tokens"],
            api_key= os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    