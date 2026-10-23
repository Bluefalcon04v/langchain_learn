from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from llm_client import get_llm
from utils.helper import print_separator, print_title

PARAMETER_DEMOS = [
    {
        "title": "Temperature",
        "description":(
            "Controls randomness. Lower values generate more deterministic responses, while higher values generate more creative responses."
        ),
        "params": {"temperature": 0.1},
        "prompt": "Suggest three AI startup ideas."
    },
    {
        "title": "Max Tokens ",
        "description":(
            "Limits the maximum number of tokens the model can generate"
        ),
        "params": {"max_tokens": 100},
        "prompt": "Explain me the Langchain "
    },
    {
        "title": "Top P",
        "description":(
            "Controls nucleus sampling by selecting tokens form the most probable candidates."
        ),
        "params": {"top_p": 0.5},
        "prompt": "Suggest three AI startup ideas"
    },
    {
        "title": "Frequency Penalty",
        "description":(
            "Reduces the repetition by discouraging the model form using same words repeatedly"
        ),
        "params": {"frequency_penalty": 1.2},
        "prompt": "Write a short paragraph about AI."
    },
    {
        "title": "Presence Penalty",
        "description":(
            "Encourages the model to introduce new idea and vocab"
        ),
        "params": {"presence_penalty": 1.4},
        "prompt": "Suggest future technologies for smart cities."
    },
    {
        "title": "Stop Sequence",
        "description":(
            "Stops text generation when a specified sequence is encountered"
        ),
        "params": {"stop": ["4."]},
        "prompt": "List five popular programming languages as numbered list."
    },
]

def main():
    print("Model with Parameters \n\n")
    
    llm = get_llm()
    
    for demo in PARAMETER_DEMOS:
        print(f"\n {demo['title']} \n")
        print(f"\n {demo['prompt']} \n")
        response = llm.bind(**demo["params"]).invoke(demo["prompt"])

        print_separator()
        print(response.content)
        print_separator()
        
if __name__ == "__main__":
    main()