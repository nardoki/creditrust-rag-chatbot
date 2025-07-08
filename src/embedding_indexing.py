import polars as pl
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import os
import pickle
import numpy as np
from tqdm import tqdm

# === Load CSV ===
df = pl.read_csv("../data/filtered/filtered_complaints.csv")
print("Loaded", df.shape[0], "rows.")

# === Setup Text Splitter ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

# === Load Model ===
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# === Chunk Texts and Collect Metadata ===
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

# === Prepare save paths ===
os.makedirs("vector_store", exist_ok=True)
embedding_path = "vector_store/embeddings.npy"
metadata_path = "vector_store/metadatas.pkl"

# === Load previous embeddings & metadata if exist ===
if os.path.exists(embedding_path) and os.path.exists(metadata_path):
    embeddings = np.load(embedding_path).tolist()
    with open(metadata_path, "rb") as f:
        metadatas = pickle.load(f)
    start_idx = len(embeddings)
    print(f"Resuming encoding from chunk {start_idx} / {len(texts)}")
else:
    embeddings = []
    start_idx = 0

# === Encode remaining chunks in batches with incremental saving ===
BATCH_SIZE = 512
for i in tqdm(range(start_idx, len(texts), BATCH_SIZE), desc="Encoding Batches"):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
    embeddings.extend(batch_embeddings)

    # Save embeddings and metadata after each batch
    np.save(embedding_path, np.array(embeddings).astype("float32"))
    with open(metadata_path, "wb") as f:
        pickle.dump(metadatas, f)

# Convert embeddings to NumPy array for FAISS
embeddings_np = np.array(embeddings).astype("float32")

# === Create FAISS index ===
dim = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings_np)

# === Save FAISS index ===
faiss.write_index(index, "vector_store/faiss_index.idx")

print("FAISS index saved successfully.")
