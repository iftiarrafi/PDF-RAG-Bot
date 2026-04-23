import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver 
import streamlit as st


st.set_page_config(
    page_title="PDF Bot",
    page_icon="📄🤖",
    layout="centered"
)

load_dotenv()

# session states
if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []  


def process_document(path):
    # Loading PDF
    loader = PyPDFLoader(path)
    docs = loader.load()

    # Splitting 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Embeddings and Vector DB
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=768
    )
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    vector_db = InMemoryVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings
    )


    @tool
    def retrieve_context(query: str):
        """Searches the uploaded documents for relevant information. Use this 
        for any questions requiring specific details from the provided files.
        """
        context = ""
        results = vector_db.similarity_search(query=query, k=3)
        for doc in results:
            context += doc.page_content + "\n\n"
        return context

    
    system_prompt = """
    You are a versatile and intelligent Document Assistant. You have access to a specific knowledge base from uploaded files.

    ### Guidelines:
    1. **Primary Source:** For any questions about the uploaded content, always use the `retrieve_context` tool. 
    2. **General Knowledge:** If the user asks general questions unrelated to the document, use your internal knowledge to provide helpful information.
    3. **Contextual Awareness:** If you use your own knowledge rather than the document, clarify that the info is general and not from the file.
    4. **Tone:** Be professional, conversational, and clear.

    ### Constraints:
    - If specific data is NOT in the document, do not invent it. State that the document doesn't mention it.
    """


    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=st.session_state.memory
    )
    
    st.session_state.agent = agent
    st.session_state.document_uploaded = True

# USER INTERFACE
st.title("📄🤖 PDF AI Assistant")
st.markdown("""
Upload any document (Research papers, Contracts, Manuals, etc.) and chat with it in real-time.
---
""")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.session_state.document_uploaded:
        st.success("Document Loaded")
        if st.button("Reset & Clear Files"):
            st.session_state.document_uploaded = False
            st.session_state.messages = []
            st.rerun()

# Upload Section
if not st.session_state.document_uploaded:
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        with st.status("Reading documents...") as status:
            doc_path_dir = "./doc_files/"
            os.makedirs(doc_path_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(doc_path_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                process_document(file_path)
                st.write(f"Analyzed: {uploaded_file.name}")
            
            status.update(label="Ready to Chat!", state="complete", expanded=False)
        st.rerun()

# Chat UI
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    query = st.chat_input("Ask a question about your documents...")
    if query: 
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        
        with st.spinner("Thinking..."):
            response = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                {"configurable": {"thread_id": "1"}}
            )
            
            answer = response["messages"][-1].content
            st.chat_message("ai").markdown(answer)
            st.session_state.messages.append({"role": "ai", "content": answer})