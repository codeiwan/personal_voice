import re


def contains_email_context(text: str) -> bool:
    EMAIL_HINTS = [
        "이메일", "메일", "메일 주소", "전자우편",
        "골뱅이", "고병이", "@", "닷컴", "점컴", "점콤", "닷 콤",
        "com", "net", "org", "co.kr", ".com",
        "gmail", "naver", "daum", "hotmail", "kakao",
        "엣", "컴", "닷넷", "닷 넷",
        "golbaengi ", "golbaengi", 
    ]

    count = sum(hint in text.lower() for hint in EMAIL_HINTS)
    return count >= 2


def normalize_email_phrases(text: str) -> str:

    if contains_email_context(text):
        text = text.lower()
        text = text.replace("골뱅이", "@").replace("golbaengi", "@")

        domain_map = {
            "네이버": "naver",
            "지메일": "gmail",
            "구글": "gmail",
            "다음": "daum",
            "카카오": "kakao",
            "핫메일": "hotmail",
            "넷": "net",
            "골뱅이": "@",
            "golbaengi": "@",
            "고병이": "@"
        }

        for kor, eng in domain_map.items():
            text = text.replace(kor, eng)

        text = re.sub(r"(점콤|점컴|닷컴|닷 콤)", ".com", text)

        email_pattern = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_pattern:
            clean_email = email_pattern.group(0)
            return clean_email
        
    return text
