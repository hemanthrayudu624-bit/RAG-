"""
RAG Project using Generative AI (Google Gemini) + Streamlit
-------------------------------------------------------------
Pipeline:
1. Upload PDF(s)
2. Extract & chunk text
3. Embed chunks with Google Generative AI embeddings
4. Store in FAISS vector store
5. Retrieve relevant chunks for a user query
6. Generate a grounded answer using Gemini (via LangChain)
"""

import streamlit as st
import os
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS


st.set_page_config(page_title="RAG with Gemini", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vector_store(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_answer(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question, k=4)
    context = "\n\n".join(d.page_content for d in docs)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = f"""Answer the question as detailed as possible using ONLY the provided context.
If the answer is not available in the context, say "Answer is not available in the context."
Do not make up information.

Context:
{context}

Question:
{user_question}

Answer:"""
    response = model.invoke(prompt)
    return response.content, docs


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🤖 RAG Project using Generative AI (Gemini)")
st.caption("Upload PDFs, build a knowledge base, and chat with your documents.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Google API Key", type="password")
    pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Process Documents", use_container_width=True):
        if not api_key:
            st.error("Enter your Google API key.")
        elif not pdf_docs:
            st.error("Upload at least one PDF.")
        else:
            with st.spinner("Reading and indexing documents..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks, api_key)
                st.session_state.processed = True
            st.success(f"Processed {len(chunks)} chunks. You can ask questions now.")

if "history" not in st.session_state:
    st.session_state.history = []

st.divider()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Source chunks"):
                for i, d in enumerate(msg["sources"]):
                    st.markdown(f"**Chunk {i + 1}:** {d.page_content[:400]}...")

user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    if not api_key:
        st.error("Enter your Google API key in the sidebar.")
    elif not os.path.exists("faiss_index"):
        st.error("Process your documents first.")
    else:
        st.session_state.history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Thinking..."):
            answer, sources = get_answer(user_question, api_key)

        st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Source chunks"):
                for i, d in enumerate(sources):
                    st.markdown(f"**Chunk {i + 1}:** {d.page_content[:400]}...")
