import streamlit as st
import tempfile

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Conversational RAG Assistant", page_icon="📄")
st.title("📄 Conversational RAG with PDF & Chat History")
st.caption("Upload PDFs and ask contextual questions using Groq + LangChain")

# ---------------- API KEY HANDLING ----------------
api_key = None

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = st.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    st.warning("Please provide a Groq API key.")
    st.stop()

# ---------------- LLM ----------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# ---------------- SESSION CHAT MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# ---------------- VECTOR STORE CREATION ----------------
@st.cache_resource
def build_vectorstore(_documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(_documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="rag_collection"
    )

    return vectorstore

# ---------------- PROCESS PDFS ----------------
if uploaded_files:

    documents = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        documents.extend(docs)

    vectorstore = build_vectorstore(documents)
    retriever = vectorstore.as_retriever()

    # ---------------- PROMPT ----------------
    system_prompt = """
You are a helpful assistant.

Answer the question using only the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ]
    )

    # ---------------- FORMAT DOCS ----------------
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ---------------- RAG CHAIN ----------------
    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # ---------------- DISPLAY OLD MESSAGES ----------------
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # ---------------- CHAT INPUT ----------------
    user_input = st.chat_input("Ask a question about your PDFs")

    if user_input:

        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        st.chat_message("user").write(user_input)

        response = conversational_rag.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": "default_session"}}
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        st.chat_message("assistant").write(response)

else:
    st.info("Upload at least one PDF to start chatting.")