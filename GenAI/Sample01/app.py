from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model=genai.GenerativeModel("gemini-pro")
def get_gemini_response(prompt):
    response = model.generate_content(prompt)
    return response.text


st.set_page_config(page_title="Gen AI Sample 01", layout="wide")
st.header("Gen AI Q&A Application")
input=st.text_input("Enter your question",key="input")
submit=st.button("Ask the question")
if submit:
    response=get_gemini_response(input)
    st.subheader("The Response is :")
    st.write(response)


