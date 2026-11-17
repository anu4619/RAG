from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summerizes text."),
    ("human", "{data}"),
])
model = ChatOpenAI(
    model="gpt-6-luna",
    temperature=1
)

