from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# llm = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0
# )

model = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4.1-Flash", temperature= 1)

llm = ChatHuggingFace(llm = model)


embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful YouTube video assistant.

Answer the question using the given transcript context.
If you don't find the answer in the context, say you don't know.

Context:
{context}

Question:
{question}
""")

summary_prompt = ChatPromptTemplate.from_template("""
Summarize the following YouTube transcript in simple and clear language.

Include the main topic, important points, and key takeaways.

Transcript:
{text}
""")


def extract_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["youtube.com", "www.youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None


def process_video(url):
    video_id = extract_video_id(url)

    if not video_id:
        raise ValueError("Please enter a valid YouTube URL")

    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id)

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )

    if not transcript:
        raise ValueError("Transcript not found")

    document = Document(page_content=transcript)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents([document])

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    summary_chain = summary_prompt | llm

    return rag_chain, summary_chain, transcript, len(chunks),retriever