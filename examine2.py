import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DB_PATH = ".\\chroma_db"

EMBEDDING_MODEL = "pkshatech/GLuCoSE-base-ja"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

COLLECTION_NAME="main_document_collection"

vector_store = Chroma(
    COLLECTION_NAME,
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

# Chromaデータベースに格納されているベクトルデータの数を取得し、コレクション名をキーとした辞書で返す
def get_num_docs():
    info_dict = {}
    for collection in vector_store._client.list_collections():
        ids = collection.get()["ids"]
        info_dict[collection.name] = len(ids)
    return info_dict

def vector_search(query, k):
    docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    l2d = [
        [doc[0].page_content] 
        + [doc[1]]
        + [json.dumps(doc[0].metadata, ensure_ascii=False)] 
        for doc in docs
    ]
    return docs

print(get_num_docs())
print(vector_search('戸賀さん', 10))