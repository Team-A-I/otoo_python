from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import anthropic
import logging

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Retrieve API key from environment variables
API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Initialize the Claude client
client = anthropic.Anthropic(api_key=API_KEY)

# 미리 정의된 프롬프트
PREDEFINED_PROMPTS = {
    "test": " 너는 이제 대화내용을 보고 누가 더 MBTI중에 'T'스러운지 너가 더 서운하게 했는지 누가 더 눈치가 없었는지 를 %로 나타내주고 대화내용에서 누가 더 과실비율이 큰지 비율을 나타내주고 둘사이의 절충안을 '기쁨''슬픔''화남'의 성격을 가진 캐릭터별로 절충안을 내어줘 마지막으로 대화내용중에 각자 대화내용중에 생각하는 키워드를 top5결과를 내어줘. 단 여기서 단어별 키워드로 나타내줘 ",
}

class PromptRequest(BaseModel):
    prompt_type: str
    chat_content: str

@app.get("/fastapi-endpoint")
def read_fastapi():
    return {"message": "Hello from FastAPI"}

@app.post("/predefined-prompt")
async def call_llm_with_predefined_prompt(request: PromptRequest):
    try:
        logging.info(f"Request data: {request}")

        if request.prompt_type not in PREDEFINED_PROMPTS:
            raise HTTPException(status_code=400, detail="Invalid prompt type")

        prompt = PREDEFINED_PROMPTS[request.prompt_type] + request.chat_content

        messages = [
            {'role': 'user', 'content': prompt}
        ]

        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            messages=messages,
            max_tokens=4000
        )
        return response
    except Exception as err:
        logging.error(f"An error occurred: {err}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {err}")
