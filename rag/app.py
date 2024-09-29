import os
import streamlit as st
import chromadb
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_chroma import Chroma
import google.generativeai as genai

# innitialize the generative model
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# innitialize the chromadb client
chroma_client = chromadb.HttpClient(host=os.environ.get("CHROMADB_HOST"), port=os.environ.get("CHROMADB_PORT")
, ssl=False)
chroma_client.heartbeat()
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def create_prompt(question, document):
    return f"You are a helpful and informative bot that answers questions using text from the reference passage included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
strike a friendly and converstional tone. \
If the passage is irrelevant to the answer, you may ignore it. \
QUESTION: '{question}' \
PASSAGE: '{document}'"

def main():

    st.title("simple RAG app")
    st.warning("Upload a PDF file to get started!")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # print chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # handle the file upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf", accept_multiple_files=False
    )
    if uploaded_file is not None:
        tempdir = tempfile.mkdtemp()
        path = os.path.join(tempdir, uploaded_file.name)
        with open(os.path.join(tempdir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        # st.write(splits)
        Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            collection_name="test",
            client=chroma_client,
        )
        collection = chroma_client.get_collection(name="test")
    # get the question
        if prompt := st.chat_input("What do you want to say to your PDF?"):
            with st.chat_message("user"):
                st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
            relevent_document = collection.query(query_texts=[prompt], n_results=1)[
                "documents"
            ][0][0]
            answer = model.generate_content(create_prompt(prompt, relevent_document)).text
            with st.chat_message("bot"):
                st.markdown(answer)
                st.session_state.messages.append({"role": "bot", "content": answer})


if __name__ == "__main__":
    main()
