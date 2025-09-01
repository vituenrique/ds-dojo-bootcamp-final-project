import streamlit as st
import utils
from streaming import StreamHandler

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header("Basic Chatbot (No Memory)")
st.write("Each reply is based only on your latest message.")

# Optional: keep UI transcript for display only
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # [{role, content}]

utils.configure_openai_api_key()

def build_chain(model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0, streaming=True)
    # NOTE: no MessagesPlaceholder, no history
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )
    return prompt | llm

if "chat_chain_nomem" not in st.session_state:
    st.session_state.chat_chain_nomem = build_chain()

@utils.enable_chat_history
def main():
    chain = st.session_state.chat_chain_nomem
    user_query = st.chat_input("Ask me anything!")
    if not user_query:
        return

    # UI transcript (does not affect model context)
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        try:
            placeholder = st.empty()
            st_cb = StreamHandler(placeholder)

            # Invoke without memory; only current input is sent
            result = chain.invoke({"input": user_query}, config={"callbacks": [st_cb]})

            # For UI transcript only
            st.session_state.messages.append(
                {"role": "assistant", "content": result.content}
            )
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

