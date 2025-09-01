# 🚀 Python → Streamlit Conversion Guide

## Overview

This guide shows you how to convert a plain Python script—or a few Jupyter‑notebook cells—into an interactive **Streamlit** app. It also lists every package version used in the **Data Science Dojo LLM Bootcamp** labs so you can reproduce the environment precisely.

---

## Quick‑Start

```bash
# 1  Create & activate a fresh virtual environment, then install dependencies
pip install -r requirements.txt

# 2  Launch the Streamlit demo hub (Home.py is the entry point)
streamlit run Home.py
```

*All additional demo pages live inside **`pages/`** and appear automatically in the sidebar.*

---

## Requirements – Lab Exercises Package List

> Copy this block into a `requirements.txt` file if you are starting from scratch.

<details>
<summary>Show package list</summary>

```text
redis==4.6.0
langchain==0.3.25
langchain-core==0.3.64
langchain-openai==0.3.21
langchain-community==0.3.24
langchain-experimental==0.3.4
langchain-text-splitters==0.3.8
langchain-anthropic==0.3.15
langgraph==0.4.8
weaviate-client==4.15.0
nbgitpuller==1.2.0
tiktoken==0.9.0
faiss-cpu==1.11.0
duckduckgo-search==8.0.2
unstructured==0.17.2
python-dotenv==1.0.0
chromadb==1.0.12
ragas==0.2.15
openai-agents==0.0.17
mcp==1.9.3
python-a2a==0.5.9
langchain-chroma==0.2.4
numexpr==2.10.2
pandas==2.3.0
Bottleneck==1.5.0
datasets==3.6.0
scipy==1.15.3
seaborn==0.13.2
matplotlib==3.10.3
streamlit==1.45.1
```

</details>

---

## Conversion Checklist

| Step  | Action                                                                           |
| ----- | -------------------------------------------------------------------------------- |
| **1** | Refactor code into **functions** so widgets can call them easily.                |
| **2** | Replace `print()` with Streamlit outputs (`st.write`, `st.markdown`, `st.code`). |
| **3** | Add **widgets** for parameters (`st.slider`, `st.text_input`, etc.).             |
| **4** | Render figures with `st.pyplot(fig)` or high‑level helpers (`st.line_chart`).    |
| **5** | Cache heavy calls using `@st.cache_data` or `@st.cache_resource`.                |
| **6** | Drop files into **`pages/`** to create new sidebar entries.                      |

> 💡 **Hint:** Paste any cell or code block into ChatGPT (or another LLM) and ask, “Convert this into Streamlit code.” It’s an easy way to scaffold the UI before polishing by hand.

---

## Worked Example – Notebook → Streamlit

Below is the **notebook version** of the prompt‑templating demo, followed by the **Streamlit page**. Both ask OpenAI for a one‑sentence travel tip.

### 1  Notebook cells (`prompt_templates_notebook.ipynb`)

```python
# 1  Imports and OpenAI key
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from getpass import getpass

openai_api_key = getpass("Enter your OpenAI API key: ")
```

```python
# 2  Prompt template
template = """
I really want to travel to {location}. What should I do there?

Respond in one short sentence.
"""

prompt = PromptTemplate(input_variables=["location"], template=template)
```

```python
# 3  Format with dynamic input
location = "Rome"  # change as you like
final_prompt = prompt.format(location=location)
print("Final prompt: \n", final_prompt)
```

```python
# 4  Invoke the model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
output = llm.invoke(final_prompt)
print("Model response: ", output.content)
```

### 2  Streamlit page *(auto‑shown in sidebar)*

```python
"""Interactive version – appears in Streamlit sidebar."""
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate

st.set_page_config(page_title="Prompt Templates with LangChain", page_icon="📝")

st.title("📝 Prompt Templates")
st.markdown(
    "Craft a **LangChain PromptTemplate**, fill in the destination, and get a one‑line tip."
)

# ── Sidebar controls ────────────────────────────────────────────────────────
st.sidebar.header("Configuration")

DEFAULT_TEMPLATE = (
    "I really want to travel to {location}. What should I do there?\n\n"
    "Respond in one short sentence."
)

template_text = st.sidebar.text_area(
    "Prompt template (must include {location}):",
    value=DEFAULT_TEMPLATE,
    height=140,
)

api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input(
    "OpenAI API key", type="password"
)
model_name = st.sidebar.selectbox(
    "Model", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"], index=0
)

# ── PromptTemplate object ───────────────────────────────────────────────────
prompt = PromptTemplate(input_variables=["location"], template=template_text)

# ── Main area ───────────────────────────────────────────────────────────────
location = st.text_input("Where would you like to travel?", value="Rome")

if st.button("Generate suggestion"):
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    final_prompt = prompt.format(location=location)
    st.markdown("**Final prompt sent to the model:**")
    st.code(final_prompt, language="text")

    llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key)
    with st.spinner("Contacting OpenAI…"):
        response = llm.invoke(final_prompt).content

    st.success("**Model response:** " + response)
else:
    st.caption(
        "Enter a destination, tweak the template if you like, then click **Generate suggestion**."
    )
```

---

## Running the Demo

```bash
streamlit run Home.py
```

Launch **Home.py** to load every page in the sidebar, then click **Prompt Templates** to open the interactive example.

---

© Data Science Dojo 2016‑2025