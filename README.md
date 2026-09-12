# YouTube RAG Chatbot

A YouTube video summarizer and question-answering chatbot built using Retrieval-Augmented Generation (RAG).

The application allows users to enter a YouTube video URL, generate a summary of the transcript, and ask questions about the video.

## Features

- Extracts transcripts from YouTube videos
- Automatically generates video summaries
- Allows users to ask questions about the video
- Uses RAG for context-based answers
- Displays relevant source transcript chunks
- Maintains chat history
- Simple and interactive Streamlit interface

## Technologies Used

- Python
- Streamlit
- LangChain
- LangChain Expression Language (LCEL)
- Groq LLM
- Hugging Face Embeddings
- FAISS
- YouTube Transcript API
- dotenv

## How It Works

1. User enters a YouTube video URL.
2. The application extracts the video transcript.
3. The transcript is divided into smaller chunks.
4. Hugging Face embeddings convert the chunks into vectors.
5. FAISS stores the vectors for similarity search.
6. Relevant transcript chunks are retrieved for each question.
7. Groq LLM generates the final answer using the retrieved context.

## Installation

Clone the repository:

```bash
git https://github.com/Gagan-Gahwai/YouTube-RAG-Chatbot