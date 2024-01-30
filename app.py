import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import fitz 
import os
import io
import cv2
import numpy as np
from langchain_community.llms import Ollama

prompt = "Ectract only name, phone no,mail id and skills and return it as json format from the following : "

def llm_respnse_for_all(text):
    st.subheader("Extracted Text from OCR :")
    st.write(text)
    st.write("Extracted Json Format :")
    prompt = "Extract the relevant data from the provided resume content into JSON format, excluding any introductory or ending lines : "
    llm = Ollama(model="llama2")
    content = str(prompt) + str(text)
    with st.spinner("Loading answer..."):
        answer = llm.invoke(content)
    return answer

def extract_text_from_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_gray = img_array
    else:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)  

    threshold_value = 200
    _, mask = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_and(img_gray, mask)

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(img, cls=True)
    with st.spinner('Loading...'):
        text = ""
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                text += "".join(line[1][0])
    answer = llm_respnse_for_all(text)

def extract_text_from_text_file(text_file):
    with st.spinner("Processing text file..."):
        text = text_file.read()
    answer = llm_respnse_for_all(text)

    return answer

def extract_text_from_pdf(pdf_file):
    with st.spinner("Extracting text from PDF..."):
        temp_pdf_path = "temp.pdf"
        with open(temp_pdf_path, "wb") as temp_pdf:
            temp_pdf.write(pdf_file.read())
        doc = fitz.open(temp_pdf_path)

        text = ""

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        doc.close()

    os.remove(temp_pdf_path)
    answer = llm_respnse_for_all(text)

    return answer

def main():
    st.title("Resume Parser")
    uploaded_file = st.file_uploader("Choose a file to upload", type=["jpg", "jpeg", "png", "pdf","txt"])
    if uploaded_file is not None:

        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image and processing OCR....", use_column_width=True)
            text = extract_text_from_image(image)
            st.text(text)

        elif uploaded_file.type == 'text/plain':
            answer = extract_text_from_text_file(uploaded_file)
            st.text(answer)

        elif uploaded_file.type == 'application/pdf':
            text = extract_text_from_pdf(uploaded_file)
            st.text(text)

        else:
            st.warning("Unsupported file type. Please upload an image or PDF.")
        
if __name__ == "__main__":
    main()




