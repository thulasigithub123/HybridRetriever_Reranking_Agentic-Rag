# app.py
import streamlit as st
import pickle
from pathlib import Path
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer, util, CrossEncoder

# --------------------------
# Load LLM
# --------------------------
load_dotenv()
APIKEY = os.getenv("GROQ_API")

from langchain_groq import ChatGroq
LLM = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    api_key=APIKEY
)

# --------------------------
# Helper Functions
# --------------------------

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_index(chunks):
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local("vectorDBstore")
    return vectorstore

# --------------------------
# Hybrid Retrieval Functions
# --------------------------

def load_retrievers():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.load_local("vectorDBstore", embeddings, allow_dangerous_deserialization=True)
    semantic_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":10})
    
    with open("chunks.pkl", "rb") as f:
        chunk_data = pickle.load(f)
    
    docs = [Document(page_content=chunk) for chunk in chunk_data]
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 10

    ensemble = EnsembleRetriever(retrievers=[semantic_retriever, keyword_retriever], weights=[0.9, 0.1])
    return ensemble

# Bi-encoder & Cross-encoder reranking
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def bi_encoder_rerank(query, docs, top_k=20):
    doc_texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
    query_emb = bi_encoder.encode(query, convert_to_tensor=True)
    doc_embs = bi_encoder.encode(doc_texts, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, doc_embs)[0]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

def cross_encoder_rerank(query, docs, top_k=5):
    doc_texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
    pairs = [[query, text] for text in doc_texts]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# --------------------------
# Agentic RAG Logic
# --------------------------

def retrieve_context_tool(query):
    """This acts like a retrieval tool for the agent."""
    ensemble_retriever = load_retrievers()
    ensemble_docs = ensemble_retriever.invoke(query)
    bi_docs = bi_encoder_rerank(query, ensemble_docs, top_k=20)
    cross_docs = cross_encoder_rerank(query, bi_docs, top_k=5)
    return ensemble_docs, bi_docs, cross_docs

def is_context_sufficient(context, query):
    judge_prompt = f"""
You are a retrieval judge.
Question: {query}
Context: {context}
Is the context sufficient to fully and confidently answer the question? Answer only YES or NO.
"""
    decision = LLM.invoke(judge_prompt).content.strip().upper()
    return decision == "YES"

def refine_query(query):
    refine_prompt = f"""
Rewrite the following question to retrieve more specific scientific information.
Original question: {query}
Improved query:
"""
    return LLM.invoke(refine_prompt).content.strip()

def agentic_rag(query, max_iters=2):
    current_query = query
    ensemble_docs, bi_docs, cross_docs = None, None, None

    for _ in range(max_iters):
        ensemble_docs, bi_docs, cross_docs = retrieve_context_tool(current_query)
        context = "\n\n".join([d.page_content for d in cross_docs])
        if is_context_sufficient(context, query):
            break
        current_query = refine_query(current_query)

    # Final answer using only top cross encoder chunks
    final_prompt = f"""
You are a scientific assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "I need more context."
Context:
{context}
Question:
{query}
Answer:
"""
    answer = LLM.invoke(final_prompt).content
    return answer, ensemble_docs, bi_docs, cross_docs
 
# --------------------------
# Streamlit UI
# --------------------------

st.title("Agentic - Hybrid RAG PDF Q&A App")

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully!")

    if st.button("Build Index"):
        text = load_pdf(uploaded_file)
        chunks = chunk_text(text)
        build_index(chunks)
        st.success("Vector DB and chunks.pkl created!")

# Ask Queries
query = st.text_input("Ask a question about your PDF:")

if query:
    if Path("vectorDBstore").exists() and Path("chunks.pkl").exists():
        st.info("Running Agentic RAG...")

        # Run Agentic RAG (retrieval + refinement + LLM)
        answer, ensemble_docs, bi_docs, cross_docs = agentic_rag(query, max_iters=2)

        # -----------------------------
        # Visualization Dropdowns
        # -----------------------------

        # Ensemble Retriever Chunks
        st.subheader("Top Ensemble Retriever Chunks")
        with st.expander("Show Ensemble Retriever Chunks"):
            for i, doc in enumerate(ensemble_docs, start=1):
                st.text(f"[Chunk {i}] {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")

        # Bi-encoder Reranked Chunks
        st.subheader("Top Bi-encoder Reranked Chunks")
        with st.expander("Show Bi-encoder Chunks"):
            for i, doc in enumerate(bi_docs, start=1):
                st.text(f"[Chunk {i}] {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")

        # Cross-encoder Reranked Chunks
        st.subheader("Top Cross-encoder Reranked Chunks")
        with st.expander("Show Cross-encoder Chunks"):
            for i, doc in enumerate(cross_docs, start=1):
                st.text(f"[Chunk {i}] {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")

        # LLM Answer
        st.subheader("LLM Answer (Adaptive Agentic RAG)")
        st.text(answer)

    else:
        st.warning("Please upload a PDF and build the index first.")
