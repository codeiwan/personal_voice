from memory.info_memory_store import memory_dict
from extractor.info_extractor import extract_card_info
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import ChatRequest, ChatResponse
from chains.chat_chain import get_conversation_chain, get_info_conversation_chain
import librosa
import numpy as np
import colorsys
import uuid
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chain = get_conversation_chain(req.user_id)
    result = chain.run(req.message)
    return ChatResponse(response=result)

@app.post("/info-chat", response_model=ChatResponse)
def info_chat(req: ChatRequest):
    chain = get_info_conversation_chain(req.user_id)
    result = chain.run(req.message)
    return ChatResponse(response=result)

@app.get("/extract-info")
def extract_info(user_id: str = Query(...)):
    if user_id not in memory_dict:
        return JSONResponse(content={"error": "대화 기록이 없습니다."}, status_code=404)
    
    memory = memory_dict[user_id]
    summary = memory.buffer  # "요약 + 사용자 발화 포함" 내용

    info = extract_card_info(summary)
    return JSONResponse(content=info)

@app.post("/upload-audio")
def upload_audio(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4().hex}.wav"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        y, sr = librosa.load(filename)
        pitch = librosa.yin(y, fmin=80, fmax=300).mean()
        energy = np.mean(librosa.feature.rms(y=y))
        jitter = np.std(librosa.zero_crossings(y, pad=False))

        hue = (pitch - 80) / (300 - 80) * 360
        saturation = max(0.2, min(1.0, energy * 100))
        value = max(0.3, 1.0 - min(0.7, jitter * 2))

        r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
        hex_color = '#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255))

        return JSONResponse({
            "hex_color": hex_color,
            "features": {
                "pitch": float(round(pitch, 2)),
                "energy": float(round(energy, 4)),
                "jitter": float(round(jitter, 4)),
            }
        })
    
    finally:
        os.remove(filename)
