def set_recommend_stock_RAG():
  llm = ChatOpenAI(model='gpt-4o')

  embedding = OpenAIEmbeddings(model='text-embedding-3-large')

  vectorstore = Chroma(collection_name='chroma_stock_news_1', persist_directory="./chroma_3-large", embedding_function=embedding)
  retriever = vectorstore.as_retriever(search_kwargs={'k': 15})

  system_prompt = (
    """당신은 사용자에게 맞춤형 종목 추천을 제공하는 한국 금융 전문가 어시스턴트입니다. 당신의 임무는 주어진 문맥(context)을 바탕으로 사용자의 투자 성향과 선호하는 섹터에 대한 정보를 제공하는 것입니다. 
    당신의 역할은 사용자가 제공한 정보에 맞춰, 최신 뉴스 데이터를 바탕으로 해당 사용자에게 적합한 종목을 추천하는 것입니다. 
    뉴스 데이터는 관련 섹터의 최근 트렌드와 성과를 반영해야 하며, 사용자의 투자 전략과 리스크 성향에 맞는 리포트를 만들어주세요
    코스피, 코스닥 종목을 기반으로 종목을 추천해주세요.
    만약, 주어진 문맥(context)에서 동향을 찾을 수 없다면, `주어진 정보에서 시장 동향에 대한 정보를 찾을 수 없습니다`라고 답하세요.

    사용자 정보는 다음과 같습니다.
    투자 성향: 
    투자 목표: 
    선호하는 투자 전략:
    선호 섹터:
    리스크 관리:
    사용자 선호 카테고리:
    위 정보를 바탕으로, 사용자의 투자 성향과 최근 뉴스 트렌드에 맞는 한국 종목을 추천해 주세요. 당신의 임무는 주어진 문맥(context)을 바탕으로 뉴스의 최근 동향과 관련 섹터에서 주목할 만한 최신 정보를 제공하는 것입니다.

    #사용자정보(User Info): 
    {context} 

    #추천종목:"""
  )
  prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system_prompt),
          ("human", "{input}"),
      ]
  )

  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  rag_chain = create_retrieval_chain(retriever, question_answer_chain)

  return rag_chain