from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model=genai.GenerativeModel("gemini-1.5-flash-002")
def get_gemini_response(prompt, image):
    if prompt!="":
        response = model.generate_content([prompt,image])
    else:
        response = model.generate_content(image)
    return response.text


st.set_page_config(page_title="Gen AI Sample 01", layout="wide")
st.header("Gen AI Q&A Application over Image")
input=st.text_input("Enter your question",key="input")

uploaded_file=st.file_uploader("Upload your image",type=["jpg","png","jpeg"])
image=""
submit=st.button("Ask the question")

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image",use_column_width=True)


if submit:
    response=get_gemini_response(input, image)
    st.subheader("The Response is :")
    st.write(response)


