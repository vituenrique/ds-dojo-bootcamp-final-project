# Assignment 1: Add Memory to Chatbot

## Part 1: Add Memory Imports
**Add after existing imports:**
```python
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
```

## Part 2: Add Memory Store
**Add after `utils.configure_openai_api_key()`:**
```python
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)
    def clear(self) -> None:
        self.messages = []

_hist_store: Dict[str, InMemoryHistory] = {}

def get_history_by_session(session_key: str) -> BaseChatMessageHistory:
    if session_key not in _hist_store:
        _hist_store[session_key] = InMemoryHistory()
    return _hist_store[session_key]
```

## Part 3: Update build_chain() Function
**Replace:**
```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ]
)
return prompt | llm
```

**With:**
```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)
chain = prompt | llm
return RunnableWithMessageHistory(
    chain,
    get_history_by_session,
    input_messages_key="input",
    history_messages_key="history",
)
```

## Part 4: Add Session Key
**Add after `st.session_state.chat_chain_nomem = build_chain()`:**
```python
MEMORY_SESSION_KEY = "chat-session-1"
```

## Part 5: Update Chain Invoke
**Replace:**
```python
result = chain.invoke({"input": user_query}, config={"callbacks": [st_cb]})
```

**With:**
```python
result = chain.invoke(
    {"input": user_query},
    config={
        "callbacks": [st_cb],
        "configurable": {"session_id": MEMORY_SESSION_KEY},
    },
)
```

