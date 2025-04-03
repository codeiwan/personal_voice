import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from memory.memory_store import get_memory
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7, model_name="gpt-3.5-turbo")

def get_conversation_chain(user_id: str):
    memory = get_memory(user_id)
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    return conversation
