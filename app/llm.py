from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

config = dotenv_values()


def get_llm():
    return ChatOpenAI(
        api_key=config.get("OPENROUTER_API_KEY") or config.get("api_key"),
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/free",
        temperature=0.8,
    )
