import os

from dotenv import load_dotenv
from langchain.memory import ConversationSummaryMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 요약 프롬프트 템플릿 커스터마이징
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template="""
    기존 대화 요약:
    {summary}

    새로운 대화 내용:
    {new_lines}

    다음 규칙에 따라 요약을 갱신하세요:
    - 사용자가 제공한 개인 정보만 요약하여 저장하세요.
    - 예: 직업, 직책, 전화번호, 이메일, 주소
    - 중복된 정보는 갱신하지 말고 유지하세요.
    → 감정, 인사말, AI의 반응 등은 저장하지 마세요.
    → 출력은 다음 예처럼 간단하고 명확하게 만드세요.

    예:
    직업: 소프트웨어 개발자
    직책: 대리
    전화번호: 010-1234-5678
    """
)

memory_dict = {}

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

def get_info_memory(user_id: str) -> ConversationSummaryMemory:
    if user_id not in memory_dict:
        memory_dict[user_id] = ConversationSummaryMemory(
            llm=llm,
            return_messages=True,
            memory_prompt_template=SUMMARY_PROMPT
        )
    return memory_dict[user_id]
