from langchain_community.vectorstores import FAISS


def retriever_data(question: str, embedding):
    # 로컬 DB 불러오기
    # 잠재적으로 위험한 데이터 구조나 객체를 포함할 수 있는 인덱스 파일의 로딩을 허용. 주로 자신이 직접 생성하고 저장한 인덱스 파일을 로드할 때 사용
    vector = FAISS.load_local("MY_FAISS_INDEX", embedding, allow_dangerous_deserialization=True)
    retrievers = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # 유사도 높은 5문장 추출
    retriever = retrievers.invoke(question)
    return retriever
