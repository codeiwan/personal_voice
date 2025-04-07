from memory.info_memory_store import memory_dict
from extractor.info_extractor import extract_card_info
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from models.schemas import ChatResponse, ChatRequest
from chains.chat_chain import get_info_conversation_chain

router = APIRouter()

@router.post("/info-chat", response_model=ChatResponse)
def info_chat(req: ChatRequest):
    chain = get_info_conversation_chain(req.user_id)
    result = chain.run(req.message)
    return ChatResponse(response=result)

@router.get("/extract-info")
def extract_info(user_id: str = Query(...)):
    if user_id not in memory_dict:
        return JSONResponse(content={"error": "대화 기록이 없습니다."}, status_code=404)
    
    memory = memory_dict[user_id]
    summary = memory.buffer

    info = extract_card_info(summary)
    return JSONResponse(content=info)
