from langchain_community.vectorstores import FAISS


# 저장
def vectorstore(splitter, embedding):
    vector = FAISS.from_documents(splitter, embedding=embedding)
    # 로컬에 DB 저장
    vector.save_local("MY_FAISS_INDEX")
