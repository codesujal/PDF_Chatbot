import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Page config
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("PDF Chatbot - Chat with your PDF (RAG)")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # Embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemma-4-26b-a4b-it",
        temperature=0,
        google_api_key="<YOUR_GEMINI_API_KEY>"
    )

    # Session memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    query = st.chat_input("Ask something about the PDF")

    if query:

        # Save user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # Retrieve relevant chunks
                relevant_docs = retriever.invoke(query)

                context = "\n\n".join(
                    doc.page_content for doc in relevant_docs
                )

                prompt = f"""
                Answer ONLY from the context below.
                If not found, say "Not in document".

                Context:
                {context}

                Question:
                {query}
                """

                try:
                    # LLM call
                    response = llm.invoke(prompt)

                    if isinstance(response.content, list):
                        answer = ""
                        for block in response.content:
                            if block.get("type") == "text":
                                answer += block.get("text")
                    else:
                        answer = response.content

                except Exception:
                    answer = "Fallback mode (no API response):\n\n"
                    for doc in relevant_docs[:3]:
                        answer += doc.page_content[:300] + "\n\n---\n\n"

                st.markdown(answer)

        # Save assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Debug panel (always available, collapsible)
        debug_data = {
            "question": query,
            "retrieved_chunks": [
                doc.page_content for doc in relevant_docs[:3]
            ],
            "context_used": context,
            "answer": answer
        }

        with st.expander("View RAG Debug Details", expanded=False):
            st.json(debug_data)