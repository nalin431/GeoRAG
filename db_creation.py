from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
import os
import shutil
import glob

load_dotenv(dotenv_path=".env.local")
openai.api_key = os.environ["OPENAI_API_KEY"]

Chroma_path = "chroma"
DATA_PATH_SCRIPTS = "Data/scripts"
DATA_PATH_PDFS = "Data/pdfs"

def main():
    chunks = chunk_creator()
    chroma_db(chunks)



def load_doc():
    documents = []
    
    # Load markdown files from scripts directory
    if os.path.exists(DATA_PATH_SCRIPTS):
        md_loader = DirectoryLoader(DATA_PATH_SCRIPTS, glob="*.md")
        md_documents = md_loader.load()
        documents.extend(md_documents)
        print(f"Loaded {len(md_documents)} markdown documents from {DATA_PATH_SCRIPTS}")
    
    # Load PDF files from pdfs directory
    if os.path.exists(DATA_PATH_PDFS):
        pdf_files = glob.glob(os.path.join(DATA_PATH_PDFS, "*.pdf"))
        for pdf_file in pdf_files:
            pdf_loader = PyPDFLoader(pdf_file)
            pdf_documents = pdf_loader.load()
            documents.extend(pdf_documents)
            print(f"Loaded {len(pdf_documents)} pages from {pdf_file}")
    
    return documents

def chunk_creator():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    documents = load_doc()
    if not documents:
        print("No documents found to process!")
        return []
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    # Show a sample chunk if available
    if len(chunks) > 1:
        document = chunks[1]
        print(f"\nSample chunk content (first 200 chars):")
        print(document.page_content[:200] + "...")

    return chunks

def chroma_db(chunks):
    if os.path.exists(Chroma_path):
        shutil.rmtree(Chroma_path)

    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=Chroma_path)

    db.persist()
    print(f"Chroma DB created/updated; {len(chunks)} chunks added.")

if __name__ == "__main__":
    main()