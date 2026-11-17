from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
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

retriever=vectorestore.as_retriever()

llm =OpenAI(model="gpt-4o-mini")

multi_query_retriever=MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

query="What is gradient descent?"

docs=multi_query_retriever.invoke(query)
print("retrieved documents:")
for doc in docs:
    print(doc.page_content)