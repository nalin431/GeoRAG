from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
import os
import shutil

load_dotenv(dotenv_path=".env.local")
openai.api_key = os.environ["OPENAI_API_KEY"]

Chroma_path = "chroma"
DATA_PATH = "data/scripts"

def main():
    chunks = chunk_creator()
    chroma_db(chunks)



def load_doc():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def chunk_creator():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(load_doc())


    document = chunks[1]
    print(document.page_content)

    return chunks

def chroma_db(chunks):
    if os.path.exists(Chroma_path):
        shutil.rmtree(Chroma_path)

    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=Chroma_path)

    db.persist()
    print(f"Chroma DB created/updated; {len(chunks)} chunks added.")

if __name__ == "__main__":
    main()