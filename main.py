import streamlit as st
from streamlit_chat import message
from backend.core import from_llm
from dotenv import load_dotenv

load_dotenv()

st.header("streamlit for documentation helper")
prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

if "user_input_history" not in st.session_state:
    st.session_state["user_input_history"] = []
if "bot_answer_history" not in st.session_state:
    st.session_state["bot_answer_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def create_source_string(in_source: list) -> str:
    return "".join([f"{i+1} {in_source[i]} \n" for i in range(len(in_source))])


if prompt:
    with st.spinner("Generating response.."):
        generated_response = from_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        source = list(
            set([i.metadata["source"] for i in generated_response["source_documents"]])
        )
        formatted_response = (
            f"{generated_response['answer']} \n\n{create_source_string(source)}"
        )
        # print(formatted_response)
        st.session_state["user_input_history"].append(prompt)
        st.session_state["bot_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))

if st.session_state:
    for question, answer in list(
        reversed(
            list(
                zip(
                    st.session_state["user_input_history"],
                    st.session_state["bot_answer_history"],
                )
            )
        )
    ):
        message(is_user=True, message=question)
        message(is_user=False, message=answer)
