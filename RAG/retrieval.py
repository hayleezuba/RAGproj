"""
References: 
https://www.geeksforgeeks.org/nlp/rag-architecture/
https://docs.trychroma.com/docs/collections/manage-collections
https://docs.trychroma.com/cloud/search-api/ranking 
"""
from langchain_core.documents import Document
import chromadb
from RAG.config import CHROMA_PATH, COLLECTION_NAME

# Search for k most similar chunks and return
def similarity_search(embedded_query, k=5) -> list[Document]:

    # Access collection created by ingestion.py
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # get the chunks
    similar_chunks = collection.query(
        query_embeddings=[embedded_query],
        n_results=k
    )
    # Return to main
    print(f"{k} chunks retrieved")
    return similar_chunks
