
# Step 1: getting data loaded
from langchain_community.document_loaders import PyPDFLoader
loaders = PyPDFLoader("SolarSystem_Arxiv.pdf")
doc_loader = loaders.load() 
   
# Step 2: Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=200)   # almost 900 words for 1200 tokens
chunks = splitter.split_documents(doc_loader)

import pickle
filename = 'chunked_data.pkl'
print(f"Writing chunks to {filename}...")
with open(filename, 'wb') as file_handle:
    pickle.dump(chunks,file_handle)
print("Finished writing data.")


#reading the pickle file

with open(filename, 'rb') as file_handle:
    chunk = pickle.load(file_handle)
# print(chunk[90])


# # step 3: Embedding

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
 
# # step 4: vector DB
from langchain_community.vectorstores import FAISS
DB = FAISS.from_documents(chunks,embeddings)  # for creating vector DB
DB.save_local("vectorDBstore")


