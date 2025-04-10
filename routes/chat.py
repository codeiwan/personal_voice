import io

from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import JSONResponse
from langchain.schema import HumanMessage, AIMessage

from chains.chat_chain import get_info_conversation_chain
from extractor.info_extractor import extract_card_info, is_info_complete
from memory.memory_store import memory_dict
from memory.info_memory_store import memory_dict as info_memory_dict
from models.schemas import ChatRequest, ChatResponse
from utils.stt import speech_to_text
from utils.text_normalizer import normalize_email_phrases

router = APIRouter()

# STT 변환
@router.post("/stt")
async def stt(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        audio_io = io.BytesIO(contents)
        audio_io.name = file.filename

        text_result = speech_to_text(audio_io)
        text_result = normalize_email_phrases(text_result)
        return JSONResponse(content={"text": text_result}, status_code=200)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 사용자 정보 기반 LLM 상호작용
@router.post("/info-chat", response_model=ChatResponse)
def info_chat(req: ChatRequest):
    chain = get_info_conversation_chain(req.user_id)
    result = chain.run(req.message)

    memory = info_memory_dict[req.user_id]
    conversation = "\n".join(
        f"{'USER' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" 
        for m in memory.chat_memory.messages
    )

    info = extract_card_info(conversation)
    finished = is_info_complete(info)
    if finished:
        final_message = "모든 정보를 수집했습니다. 감사합니다. 이제 명함 제작을 시작할 수 있어요!"
        return ChatResponse(response=final_message, finished=True)

    return ChatResponse(response=result)

# 사용자 정보 추출
@router.get("/extract-info")
def extract_info(user_id: str = Query(...)):
    if user_id not in info_memory_dict:
        return JSONResponse(content={"error": "대화 기록이 없습니다."}, status_code=404)
    
    memory = info_memory_dict[user_id]
    conversation = "\n".join(
        f"{'USER' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in memory.chat_memory.messages
    )

    info = extract_card_info(conversation)
    return JSONResponse(content=info)

# 사용자 정보 제거
@router.delete("/reset-memory")
def reset_memory(user_id: str = Query(...)):
    removed_general = memory_dict.pop(user_id, None)
    removed_info = info_memory_dict.pop(user_id, None)

    if removed_general or removed_info:
        return JSONResponse(content={"message": f"{user_id}님의 대화 기록이 초기화되었습니다."})
    else:
        return JSONResponse(content={"message": f"{user_id}님에 대한 기록이 존재하지 않습니다."}, status_code=404)
