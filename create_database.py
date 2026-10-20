#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv
import os
from pathlib import Path

#env file load karne ke liye
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

load_dotenv()

# test.py jis folder me hai, usi folder me 'notes.txt' dhoondhega
current_dir = Path(__file__).resolve().parent
file_path = current_dir / "document loaders" / "deeplearning.pdf"

# 1.Document loader 
loader = PyPDFLoader(str(file_path))
docs = loader.load()

#2.text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)

chunks = splitter.split_documents(docs)

#3.Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#4.Vectorstore create karna
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)