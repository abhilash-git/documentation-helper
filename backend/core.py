import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from const import INDEX_NAME


def from_llm(query: str) -> str:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_llm(
        llm=chat,
        # chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    return qa.invoke({"query": query})


if __name__ == "__main__":
    load_dotenv()
    query = "what is a langchain"
    ret = from_llm(query)
    print(ret)
