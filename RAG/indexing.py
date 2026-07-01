"""
References: 
https://docs.databricks.com/aws/en/agents/tutorials/ai-cookbook/quality-data-pipeline-rag

"""
import chromadb

client = chromadb.CloudClient(
  api_key='ck-2rpbb6AKe84jkVyqcqXSfZfsPA1Sy6AMM8F7yBjkgMRF',
  tenant='eda69300-a4c4-4f81-a19d-c0ceeb60c153',
  database='vectorDB'
)

import pandas
import numpy
# Prepare content for retrival in an efficient manner

# TODO: Split content into chunks
 # Chunking is the most important part -> directly affects model output

# Attach metadata to documents

# TODO: Embeddings for each doc

# Need indexing for both dense and sparse retrival -> RAG uses hybrid approach

# Insert embeddings into vectordb
