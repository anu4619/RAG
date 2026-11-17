
import os
import tempfile
import hashlib

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="PDF Chat - RAG",
    page_icon="📚",
    layout="wide"
)

# ---------------- SESSION STATE ----------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ---------------- MODELS ----------------

@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )


# ---------------- TEXT CLEANING ----------------

def clean_text(text):
    return text.encode(
        "utf-8",
        errors="replace"
    ).decode("utf-8")


# ---------------- PDF PROCESSING ----------------

def process_pdf(uploaded_file):

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Clean extracted text
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No readable text found in this PDF.")

        # Generate embeddings and create vector database
        embedding_model = get_embedding_model()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        return vectorstore, len(docs), len(chunks)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------- RAG RESPONSE ----------------

def get_response(query):

    vectorstore = st.session_state.vectorstore

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    # Prepare context with page numbers
    context_parts = []

    source_pages = []

    for doc in docs:

        page = doc.metadata.get("page")

        if page is not None:
            page_number = page + 1
            context_parts.append(
                f"Page {page_number}:\n{doc.page_content}"
            )
            source_pages.append(page_number)

        else:
            context_parts.append(doc.page_content)

    context = "\n\n".join(context_parts)

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a helpful AI assistant.

            Answer the question using ONLY the provided context.

            If the answer is not present in the context,
            say: "I could not find the answer in the document."

            Do not make up information.
            """
        ),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        )
    ])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate answer
    llm = get_llm()
    response = llm.invoke(final_prompt)

    return response.content, sorted(set(source_pages))


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("📚 PDF Chat")

    st.write("Upload a PDF and ask questions about its contents.")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        current_hash = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()

        # Reset old database when a different PDF is selected
        if current_hash != st.session_state.file_hash:

            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.file_hash = current_hash
            st.session_state.file_name = uploaded_file.name

    if uploaded_file is not None:

        st.caption(f"Selected: {uploaded_file.name}")

        if st.button(
            "⚡ Process PDF",
            use_container_width=True
        ):

            with st.spinner(
                "Reading PDF, splitting text and generating embeddings..."
            ):

                try:
                    vectorstore, pages, chunks = process_pdf(
                        uploaded_file
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.messages = []

                    st.success("PDF processed successfully!")

                    st.write(f"📄 Pages: {pages}")
                    st.write(f"🧩 Chunks: {chunks}")

                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

    st.divider()

    if st.session_state.vectorstore is not None:

        st.success("Document ready for questions")

        if st.button(
            "🗑️ Clear conversation",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.rerun()

    st.caption("Powered by LangChain, ChromaDB and OpenAI")


# ---------------- MAIN CHAT UI ----------------

st.title("💬 Chat with your PDF")

st.markdown(
    "Ask questions and get answers based on your uploaded document."
)

if st.session_state.vectorstore is None:

    st.info(
        "👈 Upload a PDF from the sidebar and click "
        "**Process PDF** to start chatting."
    )

    st.markdown("### How it works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Upload")
        st.write("Upload your research paper, notes or any PDF.")

    with col2:
        st.markdown("### 🧠 Process")
        st.write("The PDF is split into chunks and converted into embeddings.")

    with col3:
        st.markdown("### 💬 Ask")
        st.write("Ask questions and retrieve relevant information.")

else:

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):

                with st.expander("📖 View source pages"):
                    st.write(
                        ", ".join(
                            f"Page {p}" for p in message["sources"]
                        )
                    )

    # Chat input
    query = st.chat_input(
        "Ask something about your PDF..."
    )

    if query:

        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        # Generate assistant response
        with st.chat_message("assistant"):

            with st.spinner("Searching document and generating answer..."):

                try:
                    answer, sources = get_response(query)

                    st.markdown(answer)

                    if sources:
                        with st.expander("📖 View source pages"):
                            st.write(
                                ", ".join(
                                    f"Page {p}" for p in sources
                                )
                            )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
