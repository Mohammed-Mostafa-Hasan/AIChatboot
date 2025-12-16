# LangChain Conversational Agent with OpenAI, Retrieval, Agents, and Streamlit

This repository demonstrates how to build a **full conversational AI agent** using **LangChain** and **OpenAI-compatible models** (via OpenRouter), enhanced with **memory**, **retrieval (FAISS vector DB)**, **external tools (SerpAPI)**, and a **Streamlit frontend**.

The project walks through:

* Basic LLM usage
* Prompt templates and chains
* Conversational memory
* Retrieval-Augmented Generation (RAG)
* Agents with tools
* Streamlit-based UI

---

## 📌 High-Level Architecture

LangChain provides a layered architecture:

1. **Core / Backbone**

   * Abstractions for prompts, chains, memory, tools, agents
   * Runtime logic for orchestrating LLM calls

2. **Integrations Layer**

   * OpenAI / OpenRouter
   * FAISS vector database
   * SerpAPI search
   * PDF loaders

3. **Prebuilt Architectures**

   * `LLMChain`
   * `ConversationChain`
   * `ConversationalRetrievalChain`
   * Conversational Retrieval Agents

4. **Serving Layer**

   * Streamlit app to expose the agent as a chat UI

5. **Observability Layer**

   * `verbose=True` for debugging
   * Callback handlers for token streaming

---

## 📦 Libraries Used

### Standard Libraries

* **os** – Access environment variables and OS-level data
* **dotenv** – Load secrets from `.env` files (kept out of Git via `.gitignore`)

### LangChain Ecosystem

* `langchain-core`
* `langchain-community`
* `langchain-openai`

### LLM & AI

* **openai** – OpenAI-compatible client (used with OpenRouter)
* **tiktoken** – Token counting
* **huggingface_hub** – Optional model integrations

### Retrieval & Data

* **faiss-cpu** – Vector similarity search
* **pypdf** – PDF document loading

### Tools

* **google-search-results (SerpAPI)** – Real-time web search

### Frontend

* **streamlit** – Web-based chat interface

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_or_openrouter_key
SERPAPI_API_KEY=your_serpapi_key
```

Make sure `.env` is listed in `.gitignore`.

---

## ⚙️ Installation

```bash
pip install --upgrade \
  openai \
  tiktoken \
  langchain \
  langchain-core \
  langchain-community \
  langchain-openai \
  faiss-cpu \
  google-search-results \
  pypdf \
  huggingface_hub \
  streamlit
```

---

## 🤖 Basic OpenAI / OpenRouter Usage

The project first demonstrates a **raw OpenAI client call** using OpenRouter:

```python
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENAI_API_KEY
)
```

This verifies connectivity before moving to LangChain abstractions.

---

## 🧠 LangChain Components Overview

### PromptTemplate

Used to structure user input and conversation history into a reusable prompt.

**Benefits:**

* Reusability
* Clear separation of prompt and logic
* Easier debugging

---

### LLMChain

Connects a prompt template with an LLM.

**Why use it?**

* Modular workflows
* Easy chaining
* Cleaner code

---

### ConversationChain

Adds **memory** to the LLM so it can maintain context across turns.

```python
ConversationChain(llm=chat, memory=memory, verbose=True)
```

---

### ConversationBufferMemory

Stores the full conversation history.

```python
memory = ConversationBufferMemory()
```

Used in:

* ConversationChain
* Retrieval chains
* Agents

---

## 📚 Retrieval-Augmented Generation (RAG)

### Workflow

1. Load documents (PDF)
2. Split text into chunks
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve relevant chunks during chat

### Custom Embeddings Class

A custom `Embeddings` implementation is used to ensure compatibility with FAISS:

```python
class MyOpenAIEmbedding(Embeddings):
    def embed_documents(self, texts):
        return create_embeddings(texts)

    def embed_query(self, text):
        return create_embeddings([text])[0]
```

---

### ConversationalRetrievalChain

Combines:

* Chat model
* Vector retriever
* Conversation memory

```python
qa_chat = ConversationalRetrievalChain.from_llm(
    chat,
    retriever=db.as_retriever(),
    memory=memory,
    verbose=True
)
```

---

## 🛠️ Agents and Tools

### Retriever Tool

Allows the agent to search internal knowledge (PDFs):

```python
tool = create_retriever_tool(
    db.as_retriever(),
    "italy_travel",
    "Searches and returns documents regarding Italy."
)
```

### External Search Tool (SerpAPI)

Enables real-time web queries for current events.

---

### Conversational Retrieval Agent

The agent decides **when to use tools vs. LLM reasoning**:

```python
agent_executor = create_conversational_retrieval_agent(
    chat,
    tools,
    memory_key='chat_history',
    verbose=True
)
```

---

## 🖥️ Streamlit Frontend

The Streamlit app provides:

* Chat interface
* Streaming responses
* Session-based conversation memory
* Clear conversation button

### Run the App

```bash
streamlit run app.py
```

---

## 🔄 Streaming & Callbacks

A custom callback handler streams tokens in real time:

```python
class StreamlitCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")
```

---

## 🐞 Debugging & Verbose Mode

* `verbose=True` logs prompts, tool usage, and chain steps
* Helpful during development and testing

---

## 🚀 Features Summary

✔ Conversational memory
✔ Custom prompt templates
✔ PDF-based knowledge (RAG)
✔ FAISS vector database
✔ Tool-using agents
✔ External web search
✔ Streaming UI with Streamlit

---

## 📌 Notes

* If you encounter `NotImplementedError` related to token counting, upgrade `openai` and `tiktoken`.
* OpenRouter models are OpenAI-compatible but may differ in tokenization behavior.

---

## 📄 License

This project is for **educational and experimental purposes**.

---

## 🙌 Acknowledgments

* [LangChain Documentation](https://docs.langchain.com)
* OpenAI & OpenRouter
* FAISS
* Streamlit

