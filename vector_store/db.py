from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
docs = [
    Document(page_content="Python is widely used in data science and machine learning.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is a powerful data manipulation library in Python.", metadata={"source": "AI_book"}),
    Document(page_content="Neural networks are a key component of deep learning.", metadata={"source": "AI_book"})
]

embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")

vectorestore=Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory="chroma_db")

result = vectorestore.similarity_search("what is Python used for?",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriever = vectorestore.as_retriever()
docs = retriever.invoke("explain pandas library")

for d in docs:
    print(d.page_content)
    print(d.metadata)
