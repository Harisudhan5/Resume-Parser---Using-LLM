import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import fitz 
import os
import io
import cv2
import numpy as np
from langchain_community.llms import Ollama


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
    text = ""
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            rex+="".join(line[1][0])
    st.header("Extracted Text from OCR :")
    st.write(text)
    st.header("Extracted Json Format :")
    prompt = "Extract the relevant data from the provided resume content into JSON format, excluding any introductory or ending lines : "
    llm = Ollama(model="llama2")
    content = str(prompt) + str(text)
    answer = llm.invoke(content)
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
    st.header("Extracted Text from OCR :")
    st.write(text)
    st.header("Extracted Json Format :")
    prompt = "Extract the relevant data from the provided resume content into JSON format, excluding any introductory or ending lines : "
    llm = Ollama(model="llama2")
    content = str(prompt) + str(text)
    answer = llm.invoke(content)
    return answer

def main():
    st.title("Resume Parser")
    uploaded_file = st.file_uploader("Choose a file to upload", type=["jpg", "jpeg", "png", "pdf"])
    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            print(type(image))
            st.image(image, caption="Uploaded Image", use_column_width=True)
            text = extract_text_from_image(image)
        elif uploaded_file.type == 'application/pdf':
            text = extract_text_from_pdf(uploaded_file)
        else:
            st.warning("Unsupported file type. Please upload an image or PDF.")
            return
        if st.button("Extract Text"):
            st.subheader("Extracted Text:")
            st.text(text)

if __name__ == "__main__":
    main()
