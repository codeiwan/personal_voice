import io

from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import JSONResponse

from chains.chat_chain import get_info_conversation_chain
from extractor.info_extractor import extract_card_info
from memory.info_memory_store import memory_dict
from models.schemas import ChatRequest, ChatResponse
from utils.stt import speech_to_text

router = APIRouter()

# STT 변환
@router.post("/stt")
async def stt(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        audio_io = io.BytesIO(contents)
        audio_io.name = file.filename

        text_result = speech_to_text(audio_io)
        return JSONResponse(content={"text": text_result}, status_code=200)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 사용자 정보 기반 LLM 상호작용
@router.post("/info-chat", response_model=ChatResponse)
def info_chat(req: ChatRequest):
    chain = get_info_conversation_chain(req.user_id)
    result = chain.run(req.message)
    return ChatResponse(response=result)

# 사용자 정보 추출
@router.get("/extract-info")
def extract_info(user_id: str = Query(...)):
    if user_id not in memory_dict:
        return JSONResponse(content={"error": "대화 기록이 없습니다."}, status_code=404)
    
    memory = memory_dict[user_id]
    summary = memory.buffer

    info = extract_card_info(summary)
    return JSONResponse(content=info)
