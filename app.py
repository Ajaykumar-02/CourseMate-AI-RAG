import os
import json
import time
import uuid
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

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
base_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=base_dir / ".env")
persist_dir = str(base_dir / "chroma-db")

st.set_page_config(
    page_title="CourseMate AI | RAG Study Assistant",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


embedding_model = get_embedding_model()

# ------------------------------------------------------------------
# Privacy model
# - Deployed (default): every visitor gets their OWN Chroma collection, so
#   nobody can see anyone else's documents. Data auto-expires when idle.
# - Local: set COURSEMATE_LOCAL=1 in .env to keep one shared library and
#   save chat history to disk (single-user use on your own machine).
# ------------------------------------------------------------------
LOCAL_MODE = os.getenv("COURSEMATE_LOCAL", "0") == "1"
TTL_SECONDS = int(os.getenv("SESSION_TTL_MINUTES", "120")) * 60
REGISTRY_FILE = base_dir / "sessions.json"

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

COLLECTION = "coursemate_local" if LOCAL_MODE else f"user_{st.session_state.session_id}"


def get_vectorstore(name=None):
    return Chroma(
        collection_name=name or COLLECTION,
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )


def touch_and_purge_sessions():
    """Mark this session as active and delete data of sessions idle > TTL."""
    if LOCAL_MODE:
        return
    now = time.time()
    try:
        registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        registry = {}

    for name, last_seen in list(registry.items()):
        if name != COLLECTION and now - last_seen > TTL_SECONDS:
            try:
                get_vectorstore(name).delete_collection()
            except Exception:
                pass
            registry.pop(name, None)

    registry[COLLECTION] = now
    try:
        tmp = REGISTRY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(registry), encoding="utf-8")
        tmp.replace(REGISTRY_FILE)
    except Exception:
        pass


def get_library_docs():
    """All documents stored in the Chroma DB (the shared library)."""
    if not os.path.exists(persist_dir):
        return []
    try:
        metas = get_vectorstore().get().get("metadatas", [])
        return sorted({m["doc_name"] for m in metas if m and "doc_name" in m})
    except Exception:
        return []


def index_pdf(uploaded_file):
    """Load, split, tag and store a PDF in Chroma. Returns number of chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["doc_name"] = uploaded_file.name

    vs = get_vectorstore()
    try:  # same file uploaded again -> remove old chunks, no duplicates
        vs._collection.delete(where={"doc_name": uploaded_file.name})
    except Exception:
        pass
    vs.add_documents(chunks)
    return len(chunks)


def retrieve(question, primary, others, k_primary=5, k_other=2):
    """Primary (newest/selected) doc gets more chunks; other docs get fewer."""
    vs = get_vectorstore()

    primary_docs = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k_primary, "fetch_k": 15, "lambda_mult": 0.5,
            "filter": {"doc_name": primary},
        },
    ).invoke(question)

    other_docs = []
    if others:
        other_docs = vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k_other, "fetch_k": 10, "lambda_mult": 0.5,
                "filter": {"doc_name": {"$in": others}},
            },
        ).invoke(question)

    return primary_docs, other_docs


def build_context(primary, primary_docs, other_docs):
    parts = [f"=== PRIMARY DOCUMENT: {primary} (highest priority) ==="]
    parts += [d.page_content for d in primary_docs]
    if other_docs:
        parts.append("=== SECONDARY DOCUMENTS (lower priority, use only if needed) ===")
        parts += [f"[{d.metadata.get('doc_name', 'Unknown')}]\n{d.page_content}" for d in other_docs]
    return "\n\n".join(parts)


# ------------------------------------------------------------------
# Multi-chat state
# Each chat = its own page/workspace with its own messages and its own
# list of documents. The LAST doc in chat["docs"] is the PRIMARY doc.
# ------------------------------------------------------------------
WELCOME = (
    "Hello! I am CourseMate AI. Use the **➕ icon** in the chat box to attach a PDF, "
    "then ask me anything from it."
)


def create_chat(docs=None):
    cid = uuid.uuid4().hex[:8]
    docs = list(docs or [])
    st.session_state.chats[cid] = {
        "title": docs[-1] if docs else "New chat",
        "docs": docs,
        "messages": [{"role": "assistant", "content": WELCOME}],
    }
    st.session_state.current_chat = cid


HISTORY_FILE = base_dir / "chat_history.json"


def save_chats():
    """Write all chats to disk (local mode only; atomic write)."""
    if not LOCAL_MODE:
        return  # deployed: never write user chats to a shared file
    try:
        data = {
            "current": st.session_state.current_chat,
            "chats": st.session_state.chats,
        }
        tmp = HISTORY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(HISTORY_FILE)
    except Exception:
        pass


def load_chats():
    """Read saved chats. Returns (chats, current_id) or None if nothing saved."""
    if not LOCAL_MODE:
        return None
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        chats = data.get("chats", {})
        if not chats:
            return None
        library = set(get_library_docs())
        for c in chats.values():  # drop docs that no longer exist in Chroma
            c["docs"] = [d for d in c.get("docs", []) if d in library]
        current = data.get("current")
        if current not in chats:
            current = list(chats.keys())[-1]
        return chats, current
    except Exception:
        return None


if "chats" not in st.session_state:
    loaded = load_chats()
    if loaded:
        st.session_state.chats, st.session_state.current_chat = loaded
    else:
        st.session_state.chats = {}
        # First chat starts with whatever is already in the DB
        create_chat(docs=get_library_docs())

# Save at the start of every run (covers changes made just before st.rerun())
save_chats()
touch_and_purge_sessions()


def wipe_my_data():
    """Delete this user's documents (vectors) and chats. Runs as a button callback."""
    try:
        get_vectorstore().delete_collection()
    except Exception:
        pass
    if LOCAL_MODE:
        try:
            HISTORY_FILE.unlink()
        except Exception:
            pass
    st.session_state.chats = {}
    create_chat()
    st.session_state.confirm_wipe = False


chat = st.session_state.chats[st.session_state.current_chat]
cid = st.session_state.current_chat

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 CourseMate AI")
    st.caption("AI-Powered Study Assistant built with RAG")

    # ---- New chat / chat switcher ----
    if st.button("➕ New Chat (fresh page for a new doc)", type="primary", use_container_width=True):
        create_chat()
        st.rerun()

    chat_ids = list(st.session_state.chats.keys())
    picked = st.radio(
        "💬 Your chats",
        options=chat_ids,
        index=chat_ids.index(cid),
        format_func=lambda i: "📄 " + st.session_state.chats[i]["title"][:30],
    )
    if picked != cid:
        st.session_state.current_chat = picked
        st.rerun()

    if len(chat_ids) > 1:
        if st.button("🗑️ Delete this chat", use_container_width=True):
            del st.session_state.chats[cid]
            st.session_state.current_chat = list(st.session_state.chats.keys())[-1]
            st.rerun()

    st.divider()
    st.subheader("📚 Documents in this chat")

    include_others = True
    if chat["docs"]:
        primary_now = chat["docs"][-1]
        selected = st.selectbox(
            "Primary document (highest priority):",
            options=chat["docs"],
            index=len(chat["docs"]) - 1,
        )
        if selected != primary_now:
            chat["docs"].remove(selected)
            chat["docs"].append(selected)
            st.rerun()

        if len(chat["docs"]) > 1:
            include_others = st.checkbox(
                "Also search other docs of this chat (lower priority)",
                value=True,
                key=f"inc_{cid}",
            )
        st.success(f"🎯 Primary: **{chat['docs'][-1]}**")
    else:
        st.warning("No documents in this chat. Click ➕ in the chat box to add a PDF.")

    # ---- Reuse an already-indexed doc without re-uploading ----
    library = [d for d in get_library_docs() if d not in chat["docs"]]
    if library:
        with st.expander("📂 Add from my library"):
            to_add = st.multiselect("Already indexed documents", library, key=f"lib_{cid}")
            if st.button("Add to this chat", key=f"libbtn_{cid}") and to_add:
                chat["docs"].extend(to_add)
                if chat["title"] == "New chat":
                    chat["title"] = chat["docs"][-1]
                st.rerun()

    st.divider()
    with st.expander("🔒 Privacy & data"):
        if LOCAL_MODE:
            st.caption("Local mode: chats and documents are saved on this computer.")
        else:
            st.caption(
                "Your documents and chats are private to this browser session. "
                f"They are auto-deleted after {TTL_SECONDS // 60} minutes of inactivity."
            )
        st.checkbox("Yes, permanently delete all my chats and documents", key="confirm_wipe")
        st.button(
            "🗑️ Delete all my data",
            on_click=wipe_my_data,
            disabled=not st.session_state.get("confirm_wipe", False),
            use_container_width=True,
        )

    st.divider()
    st.markdown(
        """
    **RAG Pipeline:** PyPDFLoader → RecursiveCharacterTextSplitter →
    MiniLM Embeddings → Chroma (by `doc_name`) → MMR + Groq LLM
    """
    )

# ------------------------------------------------------------------
# Main: chat page
# ------------------------------------------------------------------
st.title("🎓 CourseMate AI")
st.caption("Click ➕ in the chat box to add a PDF to this chat, or use **New Chat** in the sidebar to start fresh on a new doc.")

for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# accept_file adds the built-in ➕ attach button (needs Streamlit >= 1.43)
prompt = st.chat_input(
    "Ask a question, or click ➕ to add a PDF...",
    accept_file="multiple",
    file_type=["pdf"],
)

if prompt:
    question = (prompt.text or "").strip()
    new_files = list(prompt.files or [])

    # 1) Newly attached PDFs -> the newest one becomes this chat's primary doc
    if new_files:
        notes = []
        with st.spinner("Processing PDF(s), generating embeddings..."):
            for f in new_files:
                n_chunks = index_pdf(f)
                if f.name in chat["docs"]:
                    chat["docs"].remove(f.name)
                chat["docs"].append(f.name)  # last = primary
                notes.append(f"📎 **{f.name}** added ({n_chunks} chunks)")

        if chat["title"] == "New chat":
            chat["title"] = chat["docs"][-1]

        note = (
            "\n\n".join(notes)
            + f"\n\n🎯 **{chat['docs'][-1]}** is now the primary document and gets the highest priority."
        )
        chat["messages"].append({"role": "assistant", "content": note})

        if not question:
            st.rerun()
        with st.chat_message("assistant"):
            st.markdown(note)

    # 2) Question
    if question:
        chat["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not chat["docs"]:
                answer_text = "Please attach a PDF first using the ➕ icon in the chat box."
                st.markdown(answer_text)
            else:
                with st.spinner("Searching documents & formulating answer..."):
                    primary = chat["docs"][-1]
                    others = chat["docs"][:-1] if include_others else []

                    primary_docs, other_docs = retrieve(question, primary, others)
                    context = build_context(primary, primary_docs, other_docs)

                    # {context}/{question} are template variables, so curly
                    # braces inside PDF text can't break the prompt.
                    prompt_template = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            "You are CourseMate AI, an academic study assistant.\n"
                            "Use ONLY the context below, retrieved from the student's documents.\n"
                            "Always prefer the PRIMARY DOCUMENT. Use SECONDARY DOCUMENTS only if "
                            "the primary document does not contain the answer, and mention which "
                            "document you used.\n"
                            "If the answer is not present in the context, reply: "
                            "'I could not find the answer in the uploaded study material.'\n\n"
                            "Context:\n{context}",
                        ),
                        ("human", "{question}"),
                    ])

                    llm = ChatGroq(
                        api_key=os.getenv("GROQ_API_KEY"),
                        model_name="openai/gpt-oss-120b",
                    )
                    response = (prompt_template | llm).invoke(
                        {"context": context, "question": question}
                    )
                    answer_text = response.content

                st.markdown(answer_text)

                with st.expander("🔍 View Retrieved Sources (Context)"):
                    for idx, doc in enumerate(primary_docs + other_docs, start=1):
                        page_num = doc.metadata.get("page", "N/A")
                        doc_source = doc.metadata.get("doc_name", "Unknown")
                        tag = "⭐ Primary" if doc_source == primary else "Secondary"
                        st.markdown(f"**Chunk {idx} ({tag} | Doc: {doc_source} | Page: {page_num})**")
                        st.caption(doc.page_content[:350] + "...")

        chat["messages"].append({"role": "assistant", "content": answer_text})
        save_chats()