import numpy as np
import pickle
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer


embedding_path = "vector_store/embeddings.npy"
metadata_path = "vector_store/metadatas.pkl"
faiss_index_path = "vector_store/faiss_index.idx"

print("Loading vector store...")
embeddings = np.load(embedding_path).astype("float32")
with open(metadata_path, "rb") as f:
    metadatas = pickle.load(f)
index = faiss.read_index(faiss_index_path)

print("Vector store loaded.")

# === Load model for embedding and generation ===
embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
#gen_model = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=256)
gen_model = pipeline("text-generation", model="gpt2", max_new_tokens=256)



PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:
"""

# === Retriever Function ===
def retrieve_top_k(question, k=2):
    q_emb = embed_model.encode([question]).astype("float32")
    distances, indices = index.search(q_emb, k)
    top_chunks = [metadatas[i]["text"] for i in indices[0]]
    return top_chunks

# # === Generator Function ===
# def generate_answer(question, context_chunks):
#     context = "\n\n".join(context_chunks)
#     prompt = PROMPT_TEMPLATE.format(context=context, question=question)
#     response = gen_model(prompt)[0]["generated_text"]
#     return response.split("Answer:")[-1].strip()

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    # Truncate prompt if too long
    max_tokens = 900  # Stay under 1024 with room for generation
    prompt = prompt[:max_tokens]

    response = gen_model(prompt)[0]["generated_text"]
    return response.split("Answer:")[-1].strip()



# === Sample Questions for Evaluation ===
sample_questions = [
    "Why was the customer upset about their credit card charges?",
    "Was there any mention of fraud in the complaint?",
    "Did the customer report issues with loan servicing?",
    "What product was the customer complaining about?",
    "Did the customer mention contacting customer service?"
]

# === Run Evaluation ===
print("\nRunning RAG pipeline on sample questions...\n")
for i, question in enumerate(sample_questions, 1):
    print(f"Q{i}. {question}")
    retrieved = retrieve_top_k(question)
    answer = generate_answer(question, retrieved)
    print("Answer:", answer)
    print("--- Retrieved Context (First 1-2):")
    for chunk in retrieved[:2]:
        print("*", chunk[:200].replace("\n", " ") + ("..." if len(chunk) > 200 else ""))
    print("\n" + "="*80 + "\n")
