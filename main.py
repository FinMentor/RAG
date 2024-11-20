import os
from typing import Union
from dotenv import load_dotenv
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel

app = FastAPI()

load_dotenv()

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embedding = OpenAIEmbeddings(model='text-embedding-3-large')

if not all([CLIENT_ID, CLIENT_SECRET]):
    raise ValueError("환경 변수가 설정되지 않았습니다.")

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    if item_id < 0:
        return {"error": "유효하지 않은 item_id입니다."}, 400
    return {"item_id": item_id, "q": q}

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
def chat(chat_message: ChatMessage):
    if len(chat_message.message) == 0:
        return {"error": "유효하지 않은 메시지입니다."}, 400

    response = generate_response(chat_message.message)
    return {"message": response}

def make_recommend_stock_answer_AI(recommend_list_data, type):
  rag_chain = set_recommend_stock_RAG(type)
  answer_list = []
  for entry in recommend_list_data:
    id = entry[0]
    input = '\n'.join([f"{key}: {value}" for key, value in entry[1].items()])
    result = rag_chain.invoke({"input" : input})
    answer = result['answer']
    answer_list.append((id, answer))

  return answer_list

def generate_response(messages):
    messages = [
        (
            "system",
            "당신은 사용자의 금융 관련 질문에 대해 전문적인 조언을 제공하는 챗봇입니다. 사용자의 질문에 대해 명확하고 유용한 답변을 해주세요.",
        ),
        ("human", messages),   
    ]
    result = llm.invoke(messages)
    return result.content