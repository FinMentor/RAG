from fastapi import FastAPI

from app.routers import api

app = FastAPI()

# API 라우터 등록
app.include_router(api.router)


@app.get("/")
async def root():
    return {"message": "Welcome to the NLP API"}
