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
    다음은 사용자와 AI가 나눈 대화 요약입니다.
    이 요약에서 명함에 필요한 정보를 JSON 형식으로만 추출해주세요.
    코드 블록(```json) 없이 순수한 JSON만 반환해주세요.

    대화 요약:
    {conversation}

    [JSON 출력 형식]
    {{
    "직업": "",
    "직책": "",
    "전화번호": "",
    "이메일": "",
    "주소": ""
    }}

    입력된 내용이 없으면 빈 문자열로 남겨도 괜찮습니다.
    """
    response = llm.predict(prompt)
    try:
        import json
        return json.loads(response)
    except Exception:
        return {"error": "JSON 변환 실패", "raw": response}
