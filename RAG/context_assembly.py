"""
Resources: https://launchdarkly.com/blog/llm-rag-tutorial/ -> prompt construction
https://www.reddit.com/r/LocalLLaMA/comments/1h6yu0u/how_is_the_rag_with_citations_at_the_end_of_each/ -> adding citations
"""
# Combine breadth, depth, and structure to decide which vectors make it into the LLMs prompt window
def build_prompt(query, results):
    chunks = results["documents"][0]
    context = "\n\n".join(chunks)
    context_prompt = f"user query: {query} context: {context} Directions: You are a career advisor. Analyze the content and generate an answer which directly answer user's query. Only use the provided information to answer. Make sure to include references to the materials you pull information from."
    return context_prompt