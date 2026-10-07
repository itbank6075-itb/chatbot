"""Offline checks: imports, dotenv parsing, and document splitting."""
from importlib.metadata import version
from io import StringIO
import sys
import warnings

warnings.simplefilter("error")

import langchain
import langchain_openai
import langchain_text_splitters
import streamlit
import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import dotenv_values, load_dotenv

if sys.version_info[:2] != (3, 11):
    raise RuntimeError("This project expects Python 3.11.")
print("Python", sys.version.split()[0])
for package in (
    "langchain", "langchain-openai", "langchain-text-splitters",
    "streamlit", "python-dotenv",
):
    print(f"{package}=={version(package)}")

chunks = RecursiveCharacterTextSplitter(
    chunk_size=40, chunk_overlap=10,
).split_documents([
    Document(page_content="Retrieval augmented generation chatbot. " * 5,
             metadata={"source": "smoke-test"})
])
if len(chunks) < 2 or any(c.metadata["source"] != "smoke-test" for c in chunks):
    raise RuntimeError("Document splitting failed.")
prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
if prompt.invoke({"question": "Hello"}).to_messages()[0].content != "Hello":
    raise RuntimeError("Prompt formatting failed.")
if dotenv_values(stream=StringIO("CHECK_VALUE=ok\n"))["CHECK_VALUE"] != "ok":
    raise RuntimeError("Dotenv parsing failed.")
print("PASS: imports, document splitting, prompt formatting, and dotenv parsing.")
print("No OpenAI API requests were made.")
