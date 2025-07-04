import polars as pl
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import os
import pickle
from tqdm import tqdm


df = pl.read_csv("../data/filtered/filtered_complaints.csv")
print("Loaded", df.shape[0], "rows.")

# ===  Set up chunking ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)
#model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")



# ===  Chunk, filter, and collect metadata ===
texts = []
metadatas = []

for row in df.iter_rows(named=True):
    complaint_id = row["complaint_id"]
    product = row["product"]
    narrative = row["narrative"]

    if len(narrative) < 100:
        continue  

    chunks = text_splitter.split_text(narrative)
    for chunk in chunks:
        texts.append(chunk)
        metadatas.append({
            "complaint_id": complaint_id,
            "product": product,
            "text": chunk
        })

print("Total chunks:", len(texts))

# ===  Generate embeddings in batches ===
BATCH_SIZE = 512
all_embeddings = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding Batches"):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
    all_embeddings.extend(batch_embeddings)

embeddings = all_embeddings

# ===  Create FAISS index ===
dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# ===  Save index and metadata ===
os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/faiss_index.idx")

with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump(metadatas, f)

print("Index and metadata saved successfully.")
