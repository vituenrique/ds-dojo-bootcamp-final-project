import os
import streamlit as st
import utils

from typing import List, TypedDict, Literal, Optional
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="ChatPDF (Simple Agentic RAG)", page_icon="📄")
st.header("Chat with your Documents — Simple Agentic RAG")
st.write("Decides between summarization or specific fact answering, then retrieves and generates accordingly.")

# --------------------------
# Utilities
# --------------------------
def save_file(file, folder="tmp") -> str:
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getvalue())
    return file_path

def build_vectorstore(files) -> FAISS:
    docs: List[Document] = []
    progress_bar = st.sidebar.progress(0, text=f"Processing {len(files)} files...")

    for idx, file in enumerate(files):
        path = save_file(file)
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
        progress_bar.progress((idx + 1) / len(files), text=f"Processed {idx+1}/{len(files)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

# --------------------------
# Simple Agentic RAG Graph
# --------------------------
class RAGState(TypedDict):
    question: str
    mode: Literal["summary", "fact"]
    documents: List[Document]
    generation: str

def build_simple_agentic_rag(retriever, llm: ChatOpenAI):
    """
    Graph:
      classify_mode -> retrieve -> generate -> END
    """

    # --- Node: classify mode (summary vs fact) ---
    SUMMARY_HINTS = ("summarize", "summary", "overview", "key points", "bullet", "synthesize")
    FACT_HINTS = ("when", "date", "who", "where", "amount", "total", "price", "figure", "specific", "exact")

    def classify_mode(state: RAGState) -> RAGState:
        q = state["question"].lower()
        if any(w in q for w in SUMMARY_HINTS) and not any(w in q for w in FACT_HINTS):
            mode: Literal["summary", "fact"] = "summary"
        elif any(w in q for w in FACT_HINTS):
            mode = "fact"
        else:
            # default to fact unless they asked to summarize
            mode = "summary" if "summary" in q or "summarize" in q else "fact"
        return {**state, "mode": mode}

    # --- Node: retrieve ---
    def retrieve(state: RAGState) -> RAGState:
        q = state["question"]
        k = 8 if state["mode"] == "summary" else 3
        docs = retriever.invoke(q)
        return {**state, "documents": docs[:k]}

    # --- Node: generate ---
    gen_prompt_summary = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant. Create a concise, faithful summary ONLY using the provided context. "
             "Prefer bullet points if helpful. Do not use outside knowledge."),
            ("human",
             "Question:\n{question}\n\n"
             "Context (multiple document chunks):\n{context}\n\n"
             "Write a grounded summary:")
        ]
    )

    gen_prompt_fact = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant. Answer precisely and ONLY using the provided context. "
             "If the context is insufficient, say so."),
            ("human",
             "Question:\n{question}\n\n"
             "Context:\n{context}\n\n"
             "Answer:")
        ]
    )

    def generate(state: RAGState) -> RAGState:
        ctx = "\n\n---\n\n".join(d.page_content for d in state.get("documents", []))
        if not ctx.strip():
            return {**state, "generation": "I couldn't find enough information in the documents to answer that."}

        if state["mode"] == "summary":
            answer = llm.invoke(gen_prompt_summary.format_messages(
                question=state["question"], context=ctx
            )).content
        else:
            answer = llm.invoke(gen_prompt_fact.format_messages(
                question=state["question"], context=ctx
            )).content
        return {**state, "generation": answer}

    # --- Build graph ---
    graph = StateGraph(RAGState)
    graph.add_node("classify_mode", classify_mode)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("classify_mode")
    graph.add_edge("classify_mode", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

# --------------------------
# App
# --------------------------
class CustomDataChatbot:
    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-4o-mini"

    def setup_graph(self, uploaded_files):
        vectordb = build_vectorstore(uploaded_files)
        retriever = vectordb.as_retriever()
        llm = ChatOpenAI(model=self.openai_model, temperature=0, streaming=False)
        return build_simple_agentic_rag(retriever, llm)
    
    @utils.enable_chat_history
    def main(self):
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        if "rag_app" not in st.session_state:
            st.session_state.rag_app = None

        uploaded_files = st.sidebar.file_uploader(
            label="Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            current = {f.name for f in uploaded_files}
            prev = {f.name for f in st.session_state.get("uploaded_files", [])}
            if current != prev or st.session_state.rag_app is None:
                st.session_state.uploaded_files = uploaded_files
                with st.spinner("Indexing and preparing…"):
                    st.session_state.rag_app = self.setup_graph(uploaded_files)
        else:
            st.error("Please upload PDF documents to continue!")
            st.stop()

        user_query = st.chat_input(placeholder="Ask for a summary or a specific fact…")
        if not user_query:
            return

        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            try:
                result = st.session_state.rag_app.invoke(
                    {"question": user_query, "mode": "fact", "documents": [], "generation": ""}
                )
                answer = result.get("generation", "").strip() or "I couldn't find enough information in the documents to answer that."
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    app = CustomDataChatbot()
    app.main()
