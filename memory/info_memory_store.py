import os

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

memory_dict = {}

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

def get_info_memory(user_id: str) -> ConversationBufferMemory:
    if user_id not in memory_dict:
        memory_dict[user_id] = ConversationBufferMemory(
            return_messages=True,
        )
    return memory_dict[user_id]
