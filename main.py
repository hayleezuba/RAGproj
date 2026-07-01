"""
References used: 
https://prashanth08.medium.com/building-your-retrieval-augmented-generation-rag-for-custom-llms-d5f95ed5ed7a
https://www.youtube.com/watch?v=GWB9ApTPTv4
https://ollama.com/library/glm-5.2

"""

from datetime import datetime
from ollama import chat

# Load Docs

# Chunk Docs

# Create Embeddings

# Get User Query
hour = datetime.now().hour

# Cute custom morning, afternoon, and evening messages
if hour > 6 and hour < 12:
    query = input("Good Morning! What would you like to know ✨")
elif hour >= 12 and hour < 18:
    query = input("Good Afternoon! What would you like to know ✨")
else:
    query = input("Hello Night Owl! What would you like to know ✨")

# Embed Query

# Similarity search with vector db

# Retrive chunks

# Feed to LLm
response = chat(
    model="llama3.2:1b",
    messages=[{'role': 'user', 'content': query}],
)
print(response.message.content)