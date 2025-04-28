import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(uploaded_files):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            
            # Use BytesIO to read the file's contents
            pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
            
            # Extract text from the PDF
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_embeddings(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    

def get_conversational_chain(question):
    prompt_template = """
    Answer the questions as detailed as possible from the provided context, making sure to 
    cite the context. Be creative and use lot of Reasoning in asking questions. 
    Provide necessary question and answers to the question in provided subject. 
    You should be able to generate objective questions with specified number of options and with one or multiple right answers specified in the question.
    Please provide summary if asked for in the mentioned number of points. If number of points is not mentioned, please provide summary in maximum 10 points.
    If the answer cannot be found in the context, say exactly "I don't know".
    Dont provide any wrong answers.
    Context:
    {context}
    Question:
    {question}
    Answer:
    """

    model= ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    chain = get_conversational_chain(user_question)
    response = chain({
        "question": user_question, 
        "input_documents": new_db.similarity_search(user_question, k=3)},
        return_only_outputs=True)
    print(response)
    st.write("Reply:", response["output_text"])
     

def main():
    st.title("Chat with PDF Documents")
    st.header("Chat with PDF Documents using Google Generative AI")
    user_question = st.text_input("Enter your question related to the PDF documents")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                
                raw_text = get_pdf_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_embeddings(text_chunks)
                st.success("Processing Completed")


             
if __name__ == "__main__":
    main()
