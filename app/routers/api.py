from fastapi import APIRouter

from app.services.embeddings import embedding_data
from app.services.loaders import loader_data
from app.services.retrievers import retriever_data
from app.services.splitters import splitter_data
from app.services.vectorstores import vectorstore

router = APIRouter()


@router.put("/dailytrendstore")
async def saveDailyTrendInfo():
    loader = loader_data()
    splitter = splitter_data(loader)
    embedding = embedding_data()
    vectorstore(splitter, embedding)


@router.get("/dailyreportsearch/{question}")
async def getDailyReportInfo(question: str):
    embedding = embedding_data()
    retriever = retriever_data(question, embedding)
    return {"retriever": retriever}
