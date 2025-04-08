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
    - 전화번호는 반드시 "010-0000-0000" 형식으로 변환
    - 이메일 주소는 주소 형식이 아닌 경우 빈 문자열로 남겨두기
    - 이메일 주소가 주어진 경우는 최대한 이메일 주소 형식에 맞춰서 작성하기
    - 입력된 정보가 없으면 빈 문자열 유지

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

    입력된 내용이 아직 없으면 빈 문자열로 남겨도 괜찮습니다.
    각 항목에 대해 사용자가 정보 제공을 거부한 경우 "null"로 응답해 주세요. 
    단순히 빈 문자열("")은 아직 응답하지 않았다는 뜻입니다.
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
        val = info.get(key, None)
        if val is None or str(val).strip() == "":
            return False
    return True
