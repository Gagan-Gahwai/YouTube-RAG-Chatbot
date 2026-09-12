import streamlit as st
from rag_pipeline import process_video

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube RAG Chatbot")
st.write("Watch a video, read its summary, and ask questions about it.")


if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "video_url" not in st.session_state:
    st.session_state.video_url = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None


with st.sidebar:

    st.header("Video Settings")

    url = st.text_input(
        "YouTube URL",
        placeholder="Paste video URL here"
    )

    process_button = st.button(
        "Process Video",
        use_container_width=True
    )

    if process_button:

        if not url:
            st.warning("Please enter a YouTube URL.")

        else:

            with st.spinner("Processing video..."):

                try:

                    rag_chain, summary_chain, transcript, chunks, retriever = process_video(url)

                    st.session_state.rag_chain = rag_chain
                    st.session_state.video_url = url
                    st.session_state.messages = []
                    # st.session_state.rag_chain = rag_chain
                    st.session_state.retriever = retriever

                    summary_response = summary_chain.invoke({
                        "text": transcript
                    })

                    st.session_state.summary = summary_response.content

                    st.success("Video processed successfully!")

                except Exception as e:

                    st.error(f"Error: {e}")


if st.session_state.video_url:

    st.video(st.session_state.video_url)

    summary_tab, chat_tab = st.tabs(["📝 Summary", "💬 Chat"])

    with summary_tab:

        st.subheader("Video Summary")

        st.write(st.session_state.summary)

    with chat_tab:

        st.subheader("Ask Questions About the Video")

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input(
            "Ask something about the video..."
        )

        if question:

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):
                 response = st.session_state.rag_chain.invoke(question)

                answer = response.content

                st.write(answer)

                # Retrieve relevant transcript chunks
                source_docs = st.session_state.retriever.invoke(question)

                with st.expander("📚 View Sources"):

                    for i, doc in enumerate(source_docs):

                        st.write(f"Source {i + 1}")

                        st.write(doc.page_content)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

else:

    st.info("Paste a YouTube URL in the sidebar to get started.")