
import pickle
#reading the pickle file
filename = 'chunked_data.pkl'
with open(filename, 'rb') as file_handle:
    chunk = pickle.load(file_handle)
# print(chunk[90])


# retrieving the semantic chunks
 
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
 

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.load_local('vectorDBstore',embeddings,allow_dangerous_deserialization=True)

semantic_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)
  



from langchain_community.retrievers import BM25Retriever
keyword_retriever = BM25Retriever.from_documents(chunk)
keyword_retriever.k = 10





from langchain_classic.retrievers import EnsembleRetriever
my_Ensemble_retriever  = EnsembleRetriever(retrievers=[semantic_retriever,keyword_retriever],weights=[0.9, 0.1])


# import os
# from dotenv import load_dotenv
# load_dotenv()

# APIKEY = os.getenv("GROQ_API")  
# from langchain_groq import ChatGroq

# LLM = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0.5,
#     api_key=APIKEY
# )

query = "what is luffy in piece"

docs = my_Ensemble_retriever.invoke(query)
print(docs)

# # bi encoder - reranking

# from sentence_transformers import SentenceTransformer, util
# import torch

# bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")


# def bi_encoder_rerank(query, docs, top_k=20):
#     query_emb = bi_encoder.encode(query, convert_to_tensor=True)

#     doc_texts = [d.page_content for d in docs]
#     doc_embs = bi_encoder.encode(doc_texts, convert_to_tensor=True)

#     scores = util.cos_sim(query_emb, doc_embs)[0]

#     ranked = sorted(
#         zip(docs, scores),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     return [doc for doc, _ in ranked[:top_k]]

# # docs_bi = bi_encoder_rerank(query, docs, top_k=20)
# # print(docs_bi)

# # cross encoder reranking

# from sentence_transformers import CrossEncoder
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# def cross_encoder_rerank(query, docs, top_k=5):
#     pairs = [[query, d.page_content] for d in docs]
#     scores = cross_encoder.predict(pairs)

#     ranked = sorted(
#         zip(docs, scores),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     return [doc for doc, _ in ranked[:top_k]]


# #final_docs = cross_encoder_rerank(query, docs_bi, top_k=5)
# # print(len(final_docs))
# # print(final_docs[0])


# from langchain_core.tools import tool

# @tool
# def retrieve_context(query):

#     """
#     Retrieve and rerank documents from the hybrid RAG system
#     based on the given query. Returns a text context for LLM.
#     """

#     docs = my_Ensemble_retriever.invoke(query)
#     docs_bi = bi_encoder_rerank(query, docs, top_k=20)
#     final_docs = cross_encoder_rerank(query, docs_bi, top_k=5)

#     context = "\n\n".join(
#         f"[Chunk {i+1}] {d.page_content}"
#         for i, d in enumerate(final_docs)
#     )

#     return context

# def is_context_sufficient(context, query):
#     judge_prompt = f"""
#         You are a retrieval judge.

#         Question:
#         {query}

#         Context:
#         {context}

#         Is the context sufficient to fully and confidently answer the question?
#         Answer only YES or NO.
#         """

#     decision = LLM.invoke(judge_prompt).content.strip().upper()
    
#     print(f"during the reasoning, my answer is {decision}")

#     return decision == "YES"


# def refine_query(query: str) -> str:
#     refine_prompt = f"""
#             Rewrite the following question to retrieve more specific scientific information.

#             Original question:
#             {query}

#             Improved query:
#             """

#     return LLM.invoke(refine_prompt).content.strip()

# def agentic_rag(query, max_iters=2):
#     current_query = query

#     for step in range(max_iters):
#         context = retrieve_context.run(current_query)

#         if is_context_sufficient(context, query):
#             break

#         current_query = refine_query(current_query)

#     final_prompt = f"""
#             You are a scientific assistant.

#             Answer the question using ONLY the context below.
#             If the answer is not present, say "I need more context ".
#             dont answer with your pretrained knowledge. just limit your answer based on the context.

#             Context:
#             {context}

#             Question:
#             {query}

#             Answer:
#             """

#     return LLM.invoke(final_prompt).content

# query = "what is luffy in piece"
# answer = agentic_rag(query)
# print(answer)

# prompt = f"you are a helpful assistant who can answer {query} based on the provided {final_docs}. you should answer only from the provided context and if you dont have enough context, so it directly that you need more context to justify"
# response = LLM.invoke(prompt)
# print(response.content)