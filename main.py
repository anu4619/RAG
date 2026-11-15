from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

chat = ChatOpenAI(
    model="gpt-6-luna",
    temperature=1
)

result = chat.invoke("hello")
print(result.content)