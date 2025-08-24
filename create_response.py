import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI


PROMPT_TEMPLATE = """
You are a helpful assistant that provides information based on the context provided.
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

---

Answer the question utilizing the context if necessary: {query}

"""




load_dotenv(dotenv_path=".env.local")
openai.api_key = os.environ["OPENAI_API_KEY"]

CHROMA_PATH = "chroma"


def main():
    parser = argparse.ArgumentParser(description="Create a response based on a query.")
    parser.add_argument("query", type=str, help="The query to create a response for.")
    args = parser.parse_args()
    query = args.query



    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

    results = db.similarity_search(query, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])


    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query)

    print(prompt)
    client = OpenAI()

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    print(response.output_text)

if __name__ == "__main__":
    main()