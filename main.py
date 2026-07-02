"""
References used: 
https://prashanth08.medium.com/building-your-retrieval-augmented-generation-rag-for-custom-llms-d5f95ed5ed7a
https://www.youtube.com/watch?v=GWB9ApTPTv4
https://ollama.com/library/glm-5.2

"""

from datetime import datetime
from ollama import chat

from RAG.retrieval import similarity_search
from RAG.context_assembly import build_prompt
from RAG.config import CHROMA_PATH, COLLECTION_NAME
from RAG.embedding import embedding_model, embed_query

# Sends prompt to LLM
def ask_llm(prompt):
    response = chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content

# Get User Query
# custom morning, afternoon, and evening messages
hour = datetime.now().hour
if hour > 6 and hour < 12:
    query = input("Good Morning! What would you like to know ✨\n")
elif hour >= 12 and hour < 18:
    query = input("Good Afternoon! What would you like to know ✨\n")
else:
    query = input("Hello Night Owl! What would you like to know ✨\n")

# Run until user is done
while query.lower() not in ["bye", "exit"]:

    # Help menu
    if query.lower() in ["help", "/help"]:
        print("Type bye to end chat")

    # Embed Query
    embedded_query = embedding_model.embed_query(query)

    # Retrive chunks
    chunks = similarity_search(embedded_query)

    # Build prompt with context
    llm_prompt = build_prompt(query, chunks)

    # Feed to LLm and output result
    response = ask_llm(llm_prompt)
    print(response)

    # Prompt another query
    query = input("What else can I answer for you?\n")