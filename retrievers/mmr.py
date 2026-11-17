from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
docs= [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning models.", metadata={"source": "ML_book"}),
    Document(page_content= "Gradient descent minimizes the loss funcion.", metadata={"source": "ML_book"}),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function.", metadata={"source": "ML_book"}),
    Document(page_content="Neural Networks use gradient descent for training.", metadata={"source": "ML_book"}),
    Document(page_content="Support Vector Machines are a type of supervised learning model.", metadata={"source": "ML_book"})
    
]

embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")

vectorestore=Chroma.from_documents(documents=docs, embedding=embedding_model)

mmr_retriever=vectorestore.as_retriever(search_type="mmr", search_kwargs={"k":3})
similarity_retriever=vectorestore.as_retriever(search_type="similarity", search_kwargs={"k":3})
print("\n======Similarity Retriever Results======\n")
similarity_docs = similarity_retriever.invoke("Explain gradient descent")
for doc in similarity_docs:
    print(doc.page_content)
    print("-" * 50)
print("\n======MMR Retriever Results======\n")

mmr_docs = mmr_retriever.invoke("Explain gradient descent")

for doc in mmr_docs:
    print(doc.page_content)
    print("-" * 50)