from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_gpt4o_client(max_tokens: int = 4096, temperature: float = 0.1):
    return ChatOpenAI(model="gpt-4o", temperature=temperature, max_tokens=max_tokens)
