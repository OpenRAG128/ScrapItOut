import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
import time
from langchain.embeddings import HuggingFaceEmbeddings


st.sidebar.title("OpenRAG")
st.sidebar.markdown(
    """
    OpenRAG is a tool that enhances the speed and efficiency of retrieving information from educational websites, 
    including the scrap it out component, allowing quick access to precise answers. 
    """
)
st.sidebar.markdown(
    """
     Whether for academic research, professional inquiries, or personal curiosity, OpenRAG's Scrap it out feature is poised 
        to revolutionize the way users engage with online educational resources. Experience the unparalleled convenience and effectiveness of Scrap it out
       – your gateway to rapid, reliable information retrieval.
    """
)

st.sidebar.markdown(
    """
    Enjoy Using Scarp it out!!
    """
    
)


st.title("Scrap it out 🦅")
st.text("")
url_link = st.text_input("Input your website link here")

# Check if website needs to be loaded (initial load or new URL)
if url_link and ("vector" not in st.session_state or url_link != st.session_state.get("loaded_url")):
    with st.spinner("Loading..."):
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = RecursiveUrlLoader(url=url_link, max_depth=10, extractor=lambda x: Soup(x, "html.parser").text)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.session_state["loaded_url"] = url_link  # Store the loaded URL
    st.success("Loaded!")

# Rest of the code for LLM and user interaction remains the same

llm = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key="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")


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions;{input}
    """
)

if url_link:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

st.text("")
query = st.text_input("Input your question here")

if query:
    start = time.process_time()
    response = (retrieval_chain.invoke({"input":query}))  
    print("Response time: ", time.process_time() - start)
    st.write(response['answer'])
    st.write("Response time: ", time.process_time() - start)

    with st.expander("NOT THE EXPECTED RESPONSE? CHECK OUT HERE"):

        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------")
