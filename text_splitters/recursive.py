from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    
    chunk_size=1000,
    chunk_overlap=10
)





data = PyPDFLoader("./document_loaders/notes.pdf")

docs = data.load()

chunks = splitter.split_documents(docs)

print(chunks[1].page_content)
    