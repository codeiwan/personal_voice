import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def extract_card_info(conversation: str) -> dict:
    prompt = f"""
    다음은 사용자와 AI가 나눈 대화 전문입니다.  
    대화에서 명함에 필요한 정보를 정확하게 추출해주세요.  
    코드 블록(```json) 없이 순수한 JSON만 반환해주세요.

    다음 형식을 꼭 지켜주세요:
    - 전화번호는 사용자의 발화 맥락에 따라 유추하여 "010-0000-0000" 형태로 스스로 변환 후 저장할것
    - 이메일 주소가 주어진 경우는 최대한 이메일 주소 형식에 맞춰서 작성하기
    - 아직 물어본 적이 없거나 입력된 정보가 없으면 "Waiting"으로 유지
    - 정보 제공을 스스로 거부한 경우 해당 항목에 "false"를 기입해 주세요.
    - 분석 상 응답을 이해하기 힘든 부분이 있을 경우 공백으로 놔두지 말고 "false"를 기입해 주세요요

    [JSON 출력 형식]
    {{
    "직업": "소프트웨어 엔지니어",
    "직급": "인턴",
    "전화번호": "010-1234-5678",
    "이메일": "hong@example.com",
    "주소": "서울특별시 강남구 테헤란로"
    }}

    대화 전문:
    {conversation}
    """

    response = llm.predict(prompt)
    try:
        import json
        return json.loads(response)
    except Exception:
        return {"error": "JSON 변환 실패", "raw": response}

REQUIRED_KEYS = ["직업", "직급", "전화번호", "이메일", "주소"]

def is_info_complete(info: dict) -> bool:
    for key in REQUIRED_KEYS:
        val = str(info.get(key, "")).strip().lower()
        if val == "waiting":
            return False
    return True