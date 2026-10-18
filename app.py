"""
DOCUMENT Q&A CHATBOT - Streamlit interface
-------------------------------------------
Run with:  streamlit run app.py
"""

import os

import streamlit as st

import rag_core

st.set_page_config(page_title="Document Q&A Chatbot", page_icon="📄")
st.title("📄 Document Q&A Chatbot")
st.caption("Upload a PDF, then ask questions. Answers come only from your document.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

# ---------- SIDEBAR: API key + upload ----------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get a free key at aistudio.google.com",
    )
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    build = st.button("Process document", type="primary", disabled=not (api_key and uploaded))

# ---------- BUILD THE INDEX ----------
if build:
    try:
        client = rag_core.get_client(api_key)
        with st.spinner("Reading and indexing your document..."):
            pages = rag_core.load_pdf(uploaded)
            if not pages:
                st.error("No text found. This may be a scanned PDF (images only).")
                st.stop()
            chunks = rag_core.chunk_pages(pages)
            st.session_state.index = rag_core.build_index(client, chunks)
            st.session_state.doc_name = uploaded.name
            st.session_state.messages = []
        st.sidebar.success(f"Ready: {len(pages)} pages, {len(chunks)} chunks")
    except Exception as e:
        st.sidebar.error(f"Something went wrong: {e}")

# ---------- CHAT ----------
if "index" not in st.session_state:
    st.info("👈 Add your API key, upload a PDF, and click **Process document**.")
    st.stop()

st.subheader(f"Chatting with: {st.session_state.doc_name}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for page, text in msg["sources"]:
                    st.markdown(f"**Page {page}**")
                    st.caption(text[:400] + ("..." if len(text) > 400 else ""))

question = st.chat_input("Ask something about the document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            client = rag_core.get_client(api_key)
            with st.spinner("Searching the document..."):
                retrieved = rag_core.retrieve(client, st.session_state.index, question)
                reply = rag_core.answer(client, question, retrieved)
            st.markdown(reply)
            sources = [(c["page"], c["text"]) for c, _ in retrieved]
            with st.expander("Sources"):
                for page, text in sources:
                    st.markdown(f"**Page {page}**")
                    st.caption(text[:400] + ("..." if len(text) > 400 else ""))
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "sources": sources}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
