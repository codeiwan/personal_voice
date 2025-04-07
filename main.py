from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from routes import chat, analyze

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from Render + FastAPI!"}


@app.get("/test/echo-number")
def echo_number(number: int = Query(..., description="숫자를 쿼리 파라미터로 전달하세요.")):
    return {
        "received_number": number,
        "message": "백엔드 GET test API가 제공한 숫자입니다."
    }

class NumberInput(BaseModel):
    number: int

@app.post("/test/echo-number")
def echo_number(data: NumberInput):
    return {
        "received_number": data.number,
        "message": "백엔드 POST test API가 제공한 숫자입니다."
    }

app.include_router(chat.router)
app.include_router(analyze.router)
