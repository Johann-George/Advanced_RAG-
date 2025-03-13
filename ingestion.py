import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# LLM
llm = ChatOllama(model="llama3.1:8b")
embeddings = OllamaEmbeddings(model="chroma/all-minilm-l6-v2-f32")
vectorstore = Chroma(embedding_function=embeddings, persist_directory=os.environ['CHROMA_PATH'])

# loader = PyPDFLoader("./refined.pdf")
loader = TextLoader("./refined.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)

all_splits = text_splitter.split_documents(docs)

_ = vectorstore.add_documents(documents=all_splits)
