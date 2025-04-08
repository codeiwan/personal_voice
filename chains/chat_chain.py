import os

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from memory.info_memory_store import get_info_memory
from memory.memory_store import get_memory

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0.7,
    model_name="gpt-4o-mini"
    )

def get_conversation_chain(user_id: str):
    memory = get_memory(user_id)
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    return conversation

def get_info_conversation_chain(user_id: str):
    memory = get_info_memory(user_id)

    with open("prompts/card_prompt.txt", "r", encoding="utf-8") as f:
        template = f.read()
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    return conversation
