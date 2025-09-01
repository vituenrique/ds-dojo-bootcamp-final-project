import streamlit as st
import utils

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import Tool

st.set_page_config(page_title="ChatWeb", page_icon="🌐")
st.header("Chatbot with Web Browser Access")

st.write("Equipped with Tavily search agent, Wikipedia, and Arxiv tools.")

class ChatbotTools:
    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-4o-mini"

    def setup_agent(self):
        # Tavily key input (unchanged)
        tavily_key = st.sidebar.text_input(
            "Tavily API Key",
            type="password",
            value=st.session_state.get("TAVILY_API_KEY", ""),
            placeholder="tvly-...",
        )
        if tavily_key:
            st.session_state["TAVILY_API_KEY"] = tavily_key

        if not st.session_state.get("TAVILY_API_KEY"):
            st.warning("Please enter your Tavily API key in the sidebar.")
            return None

        # Tavily tool (already present before)
        tavily_search = TavilySearch(
            max_results=5,
            topic="general",
            tavily_api_key=st.session_state["TAVILY_API_KEY"],
        )

        wiki_agent = Tool(
            name="wikipedia",
            func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
            description="Search Wikipedia for specific topics, people, or events.",
        )

        arxiv = Tool(
            name="arxiv",
            func=ArxivAPIWrapper().run,
            description="Search research papers, scientific articles, and preprints.",
        )

        tools = [tavily_search, wiki_agent, arxiv]

        llm = ChatOpenAI(model=self.openai_model, streaming=True)
        agent = create_react_agent(llm, tools)
        return agent

    @utils.enable_chat_history
    def main(self):
        agent = self.setup_agent()
        if not agent:
            return

        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, "user")
            with st.chat_message("assistant"):
                try:
                    placeholder = st.empty()
                    acc = ""

                    # Stream state updates (chunked by reasoning/tool steps)
                    for update in agent.stream({"messages": user_query}):
                        msgs = update.get("messages", [])
                        for m in msgs:
                            content = getattr(m, "content", "")
                            if not content and isinstance(getattr(m, "content", None), list):
                                content = "".join(
                                    c.get("text", "")
                                    for c in m.content
                                    if isinstance(c, dict) and c.get("type") == "text"
                                )
                            if content:
                                acc += content
                                placeholder.markdown(acc)

                    # Fallback if nothing streamed
                    if not acc:
                        resp = agent.invoke({"messages": user_query})
                        acc = (
                            resp["messages"][-1].content
                            if isinstance(resp, dict) and resp.get("messages")
                            else str(resp)
                        )
                        placeholder.markdown(acc)

                    st.session_state.messages.append({"role": "assistant", "content": acc})

                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    obj = ChatbotTools()
    obj.main()
