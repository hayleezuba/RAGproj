"""
Reference:
https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089 -> for chunking
geeksforgeeks.org/nlp/rag-architecture/ -> for embedding
https://www.reddit.com/r/LangChain/comments/134juu5/searching_for_a_way_to_interact_with_pdf_files/ -> langchain has built in embedding function


Data Resource: https://www.onetonline.org/find/descriptor/result/1.B.3.j
The first iteration of this project will be ran with data from O*NET about the Robotics Engineer career
"""

import chromadb
import pandas as pd
from langchain_text_splitters import CharacterTextSplitter # Chunking
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings # Embedding

# from sentence_transformers import SentenceTransformer # Embedding

# Load docs as df
doc_paths = ['Data/Structured/Engineering.csv', 'Data/Structured/RE_activities.csv', 'Data/Structured/RE_context.csv', 'Data/Structured/RE_skills.csv', 'Data/Structured/RE_tasks.csv']

# Converts the paths to pandas df, which we can then get the text from
def csv_to_text(path):
    df = pd.read_csv(path)
    return df.to_string(index=False)

# Simplistic approach used for first iteration, equally sized chunks with overlap so no information is left out
# Create a text splitter with optimal parameters outlined by databricks
text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

# Use splitter to split docs into chunks
chunked_docs = []
for doc in doc_paths:

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

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Example one from G4G, reevaluate once working

# Get metadata of chunks
texts = [doc.page_content for doc in chunked_docs]
metadatas = [doc.metadata for doc in chunked_docs]
ids = [f"chunk_{i}" for i in range(len(chunked_docs))]

# Embed chunked documents using HuggingFace
embedded_docs = embedding_model.embed_documents(texts)

# Insert embeddings into vectordb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="career_docs")

collection.add(
    ids=ids,
    documents=texts,
    embeddings=embedded_docs,
    metadatas=metadatas
)

print(f"Inserted {len(chunked_docs)} chunks into ChromaDB.")
