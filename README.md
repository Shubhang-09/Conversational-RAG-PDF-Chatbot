# Conversational RAG with PDF

## Live Demo

https://conversational-rag-pdf-chatbot-stkadgejwmmx6fjacmz9b5.streamlit.app

## Overview

This project is a Streamlit application where users can upload one or more PDF files and ask questions about them in a conversational way.

It uses a Retrieval-Augmented Generation (RAG) pipeline so that answers are generated based only on the uploaded documents. The app also maintains chat history, allowing follow-up questions within the same session.

---

## What This Project Does

- Upload multiple PDF files  
- Split documents into smaller chunks  
- Convert text into embeddings  
- Store embeddings in a vector database  
- Retrieve relevant chunks based on user query  
- Generate answers using Groq LLM  
- Maintain conversation memory for follow-up questions  

---

## Tech Used

- Python  
- Streamlit  
- LangChain (LCEL)  
- Chroma (Vector Store)  
- HuggingFace Embeddings  
- Groq API  

---

## How It Works (Simple Flow)

1. PDFs are uploaded and parsed.  
2. The text is split into chunks.  
3. Each chunk is converted into embeddings.  
4. Chroma stores the embeddings.  
5. When a question is asked:
   - Relevant chunks are retrieved.
   - Chat history is added.
   - Context + question is sent to the LLM.
6. The model generates an answer based only on retrieved content.
