from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()


# Model initialization
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b"
)


multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model
)

query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)


print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)