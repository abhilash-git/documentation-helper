import os

from Tools.scripts.objgraph import ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.readthedocs import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
testing




def ingest_docs():
    loader = ReadTheDocsLoader(path='./langchain-docs/langchain-docs/', encoding='utf-8', errors=ignore)
    raw_document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    document = text_splitter.split_documents(documents=raw_document)
    # print(document)

    for doc in document:
        old_path = doc.metadata["source"]
        new_path = old_path.replace("langchain-docs\\langchain-docs\\", 'https://api.python.langchain.com/en/latest/')
        doc.metadata["source"] = new_path

    # print(document)
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore.from_documents(document, embeddings, index_name="documentation-helper")


if __name__ == '__main__':
    ingest_docs()
