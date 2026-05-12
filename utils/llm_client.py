import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_gpt4o_client():
    """
    Initializes and returns a GPT-4o model client.
    Used for complex multimodal tasks like vision parsing and translation.
    """
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.1, # Low temperature for more deterministic outputs
        max_tokens=2000
    )