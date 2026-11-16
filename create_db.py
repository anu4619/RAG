from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

def clean_text(text):
    return text.encode(
        "utf-8", errors="replace"
    ).decode("utf-8")

loader = PyPDFLoader(str("deep_learning.pdf"))
docs = loader.load()

# Clean extracted text
for doc in docs:
    doc.page_content = clean_text(doc.page_content)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

embedding_model= OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)