<div align="center">
  <h1>🤖 PDF Chat Bot</h1>
  <p><strong>A RAG Based AI PDF reading assistant created with LangChain and Open-Source LLM models</strong></p>
  
  <!-- Badges -->
  <p>
    <a href="#"><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status"></a>
    <a href="#"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/badge/version-1.0.0-orange" alt="Version"></a>
  </p>
</div>

<br />

## 🎥 Project Demo

<div align="center">

  <a href="YOUR_UNLISTED_YOUTUBE_LINK">
    <img 
      src="https://img.youtube.com/vi/c8cNnl3ZWiA/maxresdefault.jpg" 
      alt="PDF Chat Bot Demo" 
      width="800"
    />
  </a>

  <p>
    <strong>▶️ Click the image above to watch the full demo</strong>
  </p>

</div>

# PDF Bot

PDF Bot is a Streamlit-based Retrieval-Augmented Generation (RAG) app for chatting with PDF documents. It loads uploaded PDFs, splits them into chunks, embeds the chunks into a vector index, retrieves the most relevant passages for a user query, and answers with an LLM-backed agent.

## What This Project Uses

### RAG / Orchestration
- `LangChain` for tool wiring and agent creation
- `LangGraph` `InMemorySaver` for lightweight conversational checkpointing in the session

### Frontend / App Layer
- `Streamlit` for the web UI, file upload flow, sidebar controls, and chat interface


### Document Processing
- `PyPDFLoader` from `langchain_community` to read PDF files
- `RecursiveCharacterTextSplitter` to break PDFs into overlapping chunks

### Embeddings
- `GoogleGenerativeAIEmbeddings`
- Embedding model: `gemini-embedding-2-preview`
- Output dimensionality: `768`

### Retrieval
- `InMemoryVectorStore` for storing document embeddings in memory
- Similarity search with `k=3` to fetch the top matching chunks

### Generation
- `ChatGroq`
- Model: `llama-3.3-70b-versatile`
- Temperature: `0.2`

### Environment / Utilities
- `python-dotenv` to load environment variables from `.env`
- `Makefile` shortcut for running the app

## RAG Pipeline

The app follows this pipeline:

1. The user uploads one or more PDF files through the Streamlit interface.
2. Each file is saved temporarily into `./doc_files/`.
3. `PyPDFLoader` reads the PDF and converts it into LangChain documents.
4. `RecursiveCharacterTextSplitter` splits the document into chunks with:
   - `chunk_size=1000`
   - `chunk_overlap=200`
5. Each chunk is converted into embeddings using Google's `gemini-embedding-2-preview`.
6. The embedded chunks are stored in an `InMemoryVectorStore`.
7. A LangChain tool called `retrieve_context` performs similarity search on the vector store.
8. When the user asks a question, the agent decides when to call `retrieve_context`.
9. The top relevant chunks are injected as context.
10. `llama-3.3-70b-versatile` on Groq generates the final answer.

## How the Agent Behaves

The agent is configured with a system prompt that tells it to:

- use the retrieval tool for questions about uploaded documents
- answer general questions with model knowledge when needed
- clearly distinguish between document-grounded answers and general knowledge
- avoid inventing details that are not present in the uploaded files

## Current Architecture Notes

- The vector store is in memory, so it is rebuilt at runtime and is not persisted across app restarts.
- Conversation state is stored in Streamlit session state.
- The app currently processes uploaded files one by one and rebuilds the active agent inside `process_document()`. In the current implementation, the most recently processed file becomes the active retrieval source.

## Project Structure

```text
.
|-- 5_rag_agent.py          # Main Streamlit RAG app
|-- Makefile                # Run shortcut
|-- data/                   # Sample PDFs
|-- assests/                # Project images
|-- .env                    # Local environment variables
```

## Setup

Create a `.env` file with the API keys required by the models used in the code:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

Install the required Python packages, then run:

```bash
make run
```

That starts:

```bash
streamlit run 5_rag_agent.py
```

## Why This Is RAG

This project is a RAG system because it does not rely only on the language model's built-in knowledge. Before answering document-specific questions, it retrieves relevant chunks from the uploaded PDFs and uses those chunks as grounded context for generation.
