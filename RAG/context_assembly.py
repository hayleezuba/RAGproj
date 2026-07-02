"""
Resources: https://launchdarkly.com/blog/llm-rag-tutorial/ -> prompt construction
"""
# Combine breadth, depth, and structure to decide which vectors make it into the LLMs prompt window
def build_prompt(query, results):
    chunks = results["documents"][0]
    context = "\n\n".join(chunks)
    context_prompt = f"user query: {query} context: {context} Analyze the content and generate an answer which directly answer user's query"
    return context_prompt