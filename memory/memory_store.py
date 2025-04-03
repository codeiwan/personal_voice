from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 유저 ID 별 memory 저장소 (간단한 dict 기반)
memory_dict = {}

# 요약에 사용할 LLM 준비
llm = ChatOpenAI(
    model_name = "gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def get_memory(user_id: str) -> ConversationSummaryMemory:
    if user_id not in memory_dict:
        memory_dict[user_id] = ConversationSummaryMemory(
            llm = llm,
            return_messages=True
        )
    return memory_dict[user_id]
