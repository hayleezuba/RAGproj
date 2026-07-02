"""
References used: 
https://prashanth08.medium.com/building-your-retrieval-augmented-generation-rag-for-custom-llms-d5f95ed5ed7a
https://www.youtube.com/watch?v=GWB9ApTPTv4
https://ollama.com/library/glm-5.2

"""

from datetime import datetime
from ollama import chat
from langchain_huggingface import HuggingFaceEmbeddings # Embedding


# Get User Query
# custom morning, afternoon, and evening messages
hour = datetime.now().hour
if hour > 6 and hour < 12:
    query = input("Good Morning! What would you like to know ✨")
elif hour >= 12 and hour < 18:
    query = input("Good Afternoon! What would you like to know ✨")
else:
    query = input("Hello Night Owl! What would you like to know ✨")

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Run until user is done
while (query is not None):
    query = input("What else can I answer for you?")
    
    # Embed Query
    embedded_query = embedding_model.embed_query(query)
    
    # Similarity search with vector db

    # Retrive chunks

    # Feed to LLm
    response = chat(
        model="llama3.2:1b",
        messages=[{'role': 'user', 'content': query}],
    )
    print(response.message.content)