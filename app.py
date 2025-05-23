import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import tempfile

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

uploaded_files = st.file_uploader("upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PDFPlumberLoader(tmp_path)
        documents = loader.load()
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(all_documents)

    embeddings = OllamaEmbeddings(model="gemma:2b")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="db"
    )

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve 
        relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some 
        of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )

    llm = OllamaLLM(model="gemma:2b")

    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm,
        prompt=query_prompt
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=multi_retriever,
        chain_type="stuff"
    )

    st.success("PDFs processed successfully! Ask your questions below.")

    user_question = st.text_input("Ask a question from the PDFs:")
    if user_question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(user_question)
            st.markdown(f"**Answer:** {answer}")
