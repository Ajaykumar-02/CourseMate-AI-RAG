import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Base Directory & Environment setup
base_dir = Path(__file__).resolve().parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=env_path)

persist_dir = str(base_dir / "chroma-db")

# Streamlit Page Config
st.set_page_config(
    page_title="CourseMate AI | RAG Study Assistant",
    page_icon="🎓",
    layout="wide"
)

# 1. State Management: Track if DB is ready
if "db_ready" not in st.session_state:
    # Agar folder pehle se bana hai toh ready maan sakte hain, warna False
    st.session_state.db_ready = os.path.exists(persist_dir)

# Cache embedding model
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# --- SIDEBAR: CourseMate AI Details ---
with st.sidebar:
    st.title("🎓 CourseMate AI")
    st.caption("AI-Powered Study Assistant built with RAG")
    
    st.markdown("""
    ### 📖 About the Project
    CourseMate AI eliminates the need to manually skim hundreds of textbook or lecture slide pages.

    **RAG Pipeline Stages:**
    * 📑 **Step 1:** Document Loader (`PyPDFLoader`)
    * ✂️ **Step 2:** Text Splitting (`RecursiveCharacterTextSplitter`)
    * 🔢 **Step 3:** MiniLM Embeddings
    * 🗄️ **Step 4:** Chroma Vector DB
    * 🎯 **Step 5:** MMR Retrieval + Groq LLM
    """)
    
    st.divider()
    if st.session_state.db_ready:
        st.success("🟢 Vector DB Active & Ready")
    else:
        st.warning("🔒 Chat Locked (Upload & Process PDF first)")

# --- MAIN HEADER ---
st.title("🎓 CourseMate AI")
st.markdown("Upload your study resources, build the semantic index, and chat directly with your course material.")

# ==========================================
# SECTION 1: UPLOAD MATERIAL (TOP)
# ==========================================
st.subheader("📤 1. Upload Study Material")

uploaded_file = st.file_uploader(
    label="Choose a PDF document (Book, Notes, Slides)",
    type=["pdf"]
)

if uploaded_file:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📄 File Selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
    with col2:
        create_btn = st.button("🚀 Create Vector DB", type="primary", use_container_width=True)

    if create_btn:
        with st.spinner("Processing PDF, creating chunks, and saving embeddings..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_pdf_path = tmp_file.name

            # 1. Document Loading
            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()

            # 2. Text Splitting
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            # 3. Vector DB Store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=persist_dir
            )
            
            # Unlock the chat section
            st.session_state.db_ready = True
            
            # Reset conversation for new document
            st.session_state.messages = [
                {"role": "assistant", "content": f"Document **'{uploaded_file.name}'** successfully processed! Ask me any question from it."}
            ]
            
            st.success(f"🎉 Vector DB created with **{len(chunks)} chunks**! Chat section is now unlocked below.")

st.divider()

# ==========================================
# SECTION 2: CHAT WITH MATERIAL (LOCKED UNTIL READY)
# ==========================================
st.subheader("💬 2. Chat with Your Material")

# Condition check: agar DB process nahi hua, toh chat lock rahegi
if not st.session_state.db_ready:
    st.warning("🔒 **Chat is locked:** Please upload a PDF file and click **'Create Vector DB'** above to activate the assistant.")
else:
    # Initialize message list
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am CourseMate AI. Ask me any question based on your uploaded document."}
        ]

    # Display past conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input bar (only visible when db_ready is True)
    if user_question := st.chat_input("Ask a question from your study material..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document & formulating answer..."):
                vectorstore = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embedding_model
                )

                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )

                retrieved_docs = retriever.invoke(user_question)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                system_prompt = (
                    "You are CourseMate AI, an academic study assistant.\n"
                    "Use ONLY the following context retrieved from the student's document to answer the question.\n"
                    "If the answer is not present in the context, reply: "
                    "'I could not find the answer in the uploaded study material.'\n\n"
                    f"Context:\n{context}"
                )

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{question}")
                ])

                llm = ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model_name="openai/gpt-oss-120b"
                )

                chain = prompt_template | llm
                response = chain.invoke({"question": user_question})
                answer_text = response.content

                st.markdown(answer_text)

                # Show sources inside expander
                with st.expander("🔍 View Retrieved Sources (Context)"):
                    for idx, doc in enumerate(retrieved_docs, start=1):
                        page_num = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Chunk {idx} (Page: {page_num})**")
                        st.caption(doc.page_content[:350] + "...")

                st.session_state.messages.append({"role": "assistant", "content": answer_text})