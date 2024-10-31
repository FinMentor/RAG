from langchain_text_splitters import RecursiveCharacterTextSplitter


# 분할
def splitter_data(loader):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split = splitter.split_documents(loader)
    return split
