from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

data = TextLoader(
    "document_loaders/notes.txt",
    encoding="utf-8"
)
docs = data.load()

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summerizes text."),
    ("human", "{data}"),
])
model = ChatOpenAI(
    model="gpt-6-luna",
    temperature=1
)

prompt = template.format_messages(data=docs[0].page_content)


result = model.invoke(prompt)
print(result.content)