from fastapi import FastAPI, Request, Query
from pydantic import BaseModel

app = FastAPI()

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
