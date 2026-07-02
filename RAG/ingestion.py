"""
Reference:
https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089 -> for chunking
geeksforgeeks.org/nlp/rag-architecture/ -> for embedding
https://www.reddit.com/r/LangChain/comments/134juu5/searching_for_a_way_to_interact_with_pdf_files/ -> langchain has built in embedding function


Data Resource: https://www.onetonline.org/find/descriptor/result/1.B.3.j
The first iteration of this project will be ran with data from O*NET about the Robotics Engineer career

TODO future improvements:
    -> Data
        -> Add processing of unstructured data (pdfs, websites)
        -> Includes adding a web scraper
        -> Improve code for multiple datasets (add more data cleaning and research better indexing)
    -> Chunking
        -> Research and Evaluate different chunking methods
        -> Fine tune parameters (chunk size and overlap)
    -> Embeddings
        -> Research and evaluate difference embedding models
        -> Look into embedding caching for improved response time
    -> Storage
        -> Prevent duplicates

"""

import chromadb
import pandas as pd
from langchain_text_splitters import CharacterTextSplitter # Chunking
from langchain_core.documents import Document
from RAG.config import CHROMA_PATH, COLLECTION_NAME 
from RAG.embedding import embedding_model, embed_query

# from sentence_transformers import SentenceTransformer # Embedding

# Converts the paths to pandas df, which we can then get the text from
def csv_to_text(path):
    df = pd.read_csv(path)
    return df.to_string(index=False)

def ingest_docs():

    # Load docs as df
    doc_paths = ['Data/Structured/RE_activities.csv', 'Data/Structured/RE_context.csv', 'Data/Structured/RE_skills.csv', 'Data/Structured/RE_tasks.csv']


    # Simplistic approach used for first iteration, equally sized chunks with overlap so no information is left out
    # Create a text splitter with optimal parameters outlined by databricks
    text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    # Use splitter to split docs into chunks
    chunked_docs = []
    for doc in doc_paths:
        print(f"Loading document: {doc}")

        # Chunk each doc in fixed size chunks
        text = csv_to_text(doc)
        chunks = text_splitter.split_text(text)

        # Convert chunks to Document objects
        for i, chunk in enumerate(chunks):
            # Create document object for each chunk
            doc_obj = Document(
                page_content=chunk,
                metadata={
                    "source": doc,
                    "chunk_id": f"{doc}_chunk_{i}",
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "chunk_type": "fixed-size"
                }
            )
            chunked_docs.append(doc_obj)

    # Get metadata of chunks
    texts = [doc.page_content for doc in chunked_docs]
    metadatas = [doc.metadata for doc in chunked_docs]
    ids = [doc.metadata["chunk_id"] for doc in chunked_docs]

    # Embed chunked documents using HuggingFace
    embedded_docs = embedding_model.embed_documents(texts)

    # Insert embeddings into vectordb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embedded_docs,
        metadatas=metadatas
    )

    print(f"Inserted {len(chunked_docs)} chunks into ChromaDB.")
    print(chunked_docs[0].page_content)
    print(chunked_docs[0].metadata)
    return 

def main():
    print("Before ingesting documents")
    ingest_docs()

if __name__ == '__main__':
    main()
