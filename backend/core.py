from dotenv import load_dotenv

# from langchain.chains import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from const import INDEX_NAME

load_dotenv()


def from_llm(query: str, chat_history: list):
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    # qa = RetrievalQA.from_llm(
    #     llm=chat,
    #     # chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     return_source_documents=True
    # )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    print(chat_history)
    return qa.invoke({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    # load_dotenv()
    test_history = []
    test_query = "what is a langchain"
    ret = from_llm(test_query, test_history)
    test_history.append((test_query, ret))
    print(ret)
