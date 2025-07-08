### creditrust-rag-chatbot

🚀 CrediTrust Financial Complaint AI Chatbot

**Unlock Customer Insights in Minutes — Not Days!**

🎯 What’s This Project About?
CrediTrust Financial is rapidly growing across East Africa, offering Credit Cards, Personal Loans, Buy Now Pay Later (BNPL), Savings, and Money Transfers to 500,000+ users. Every month, thousands of customer complaints pour in from multiple channels — and sorting through them is a nightmare!

This project builds a smart AI-powered chatbot that turns raw, messy complaint data into quick, clear insights. Product Managers, Support, and Compliance teams can instantly spot trends, solve problems faster, and stay ahead of customer issues — no more digging through mountains of text!

##  Features

- 🔍 Retrieves top relevant complaint excerpts using FAISS
- 💬 Generates answers using a powerful language model
- 📂 Shows source texts used to build trust
- 🧼 Clear interface built with Streamlit
- 🧠 Evaluated for quality with a curated set of real questions

---

💻 How It Works 

**Speedy Data Crunching with Polars**
Fast and lightweight processing of massive complaint datasets. No heavy memory hogging here!

**Smart Text Chunking**
Splits long complaint stories into bite-sized, meaningful pieces — ready for embedding.

**Powerful Sentence Embeddings**
Uses cutting-edge Sentence Transformers to convert text into numerical vectors that capture meaning.

**Blazing-Fast Search with FAISS**
Finds the most relevant complaint snippets lightning-fast through vector similarity search.

**Answer Generation with RAG**
Feeds retrieved info into a language model for concise, grounded, human-like answers.


**📝 Task 1: Data Exploration & Cleaning**

* Load and explore the complaint dataset.

* Analyze complaint distribution and narrative lengths.

* Filter for key products and remove empty narratives.

* Clean text by lowercasing and removing boilerplate.

* Save the cleaned dataset.


**🚀 Task 2: Chunking, Embedding & Indexing**

* Split narratives into manageable chunks.

* Embed chunks using a sentence transformer model.

* Build a FAISS index for fast similarity search.

* Save the index and metadata for later use.


## Project structure
```bash
creditrust-rag-chatbot/
├── notebooks
| ├── 01_eda_preprocessing.ipynb
├── src/
│ ├── embedding_indexing.py # Builds FAISS index 
│ ├── rag_pipeline.py # RAG logic (retriever + generator)
| ├── app.py # Streamlit web app
├── Vector_store 
└── .gitignore
└── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/creditrust-rag-assistant.git
cd creditrust-rag-assistant
```
### 2. Install Requirements

```bash
pip install streamlit sentence-transformers faiss-cpu transformers numpy polars tqdm
```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run The App

```bash
streamlit run app.py
```



