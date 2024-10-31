import bs4
from langchain_community.document_loaders import WebBaseLoader


# 로드
def loader_data():
    # BeautifulSoup : HTML 및 XML 문서를 파싱하고 구문 분석하는 데 사용되는 파이썬 라이브러리. 주로 웹 스크레이핑(웹 페이지에서 데이터 추출) 작업에서 사용되며, 웹 페이지의 구조를 이해하고 필요한 정보를 추출하는 데 유용
    loader = WebBaseLoader(web_paths=("https://www.aitimes.com/news/articleView.html?idxno=159102",
                                      "https://www.aitimes.com/news/articleView.html?idxno=159072",
                                      "https://www.aitimes.com/news/articleView.html?idxno=158943"), bs_kwargs=dict(
        # 태그
        # 태그의 ID 값들
        parse_only=bs4.SoupStrainer("article", attrs={"id": ["article-view-content-div"]})))
    load = loader.load()
    return load
