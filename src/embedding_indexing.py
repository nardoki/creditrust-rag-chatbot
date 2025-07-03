import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import pandas as pd

# Load your data
df = pd.read_csv('../data/raw/complaints.csv')

# Initialize text splitter
chunk_size = 500
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Initialize embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)

# Prepare containers for embeddings and metadata
embeddings = []
metadata = []

print("Starting chunking, embedding, and indexing...")
df.columns = df.columns.str.strip().str.lower()


for idx, row in df.iterrows():

    complaint_id = row['complaint_id']
    product = row['product']
    text = row['cleaned_text']  # Use cleaned text column

    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Embed each chunk
    chunk_embeddings = embedder.encode(chunks)
    
    for i, chunk_emb in enumerate(chunk_embeddings):
        embeddings.append(chunk_emb)
        metadata.append({
            'complaint_id': complaint_id,
            'product': product,
            'chunk_index': i,
            'text_chunk': chunks[i]
        })

print(f"Total chunks processed: {len(embeddings)}")

# Convert embeddings list to numpy array
embedding_dim = len(embeddings[0])
embedding_matrix = np.array(embeddings).astype('float32')

# Build FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(embedding_matrix)

# Save FAISS index and metadata
os.makedirs('vector_store', exist_ok=True)
faiss.write_index(index, 'vector_store/faiss_index.bin')

with open('vector_store/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Vector store saved in 'vector_store/' directory.")
