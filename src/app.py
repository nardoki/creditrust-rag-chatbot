import streamlit as st
from rag_pipeline import retrieve_top_k, generate_answer

# === Streamlit Page Config ===
st.set_page_config(page_title="CrediTrust Chatbot", layout="centered")
st.title("📊 CrediTrust Complaint Assistant")
st.markdown("Ask any question about customer complaints. The AI will respond based on real complaint data.")

# === Session State ===
if "history" not in st.session_state:
    st.session_state.history = []

# === User Input ===
question = st.text_input("Your question:", placeholder="e.g. What were common issues with credit cards?")
col1, col2 = st.columns([1, 1])

with col1:
    submit = st.button("Ask")
with col2:
    clear = st.button("Clear")

# === Handle Clear Button ===
if clear:
    st.session_state.history = []
    st.rerun()

    

# === Handle Submit Button ===
if submit and question:
    with st.spinner("Retrieving answer..."):
        retrieved_chunks = retrieve_top_k(question)
        answer = generate_answer(question, retrieved_chunks)

    st.session_state.history.append({
        "question": question,
        "answer": answer,
        "sources": retrieved_chunks[:2]  
    })

# === Display Chat History ===
for qa in reversed(st.session_state.history):
    st.markdown(f"**You:** {qa['question']}")
    st.markdown(f"**Assistant:** {qa['answer']}")
    with st.expander("🔍 View Source Chunks"):
        for i, src in enumerate(qa["sources"], 1):
            st.markdown(f"**Source {i}:**\n{src}")
#python -m streamlit run app.py
