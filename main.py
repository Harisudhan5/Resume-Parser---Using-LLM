import streamlit as st
from PIL import Image
import pytesseract

def extract_text(image):
    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(image)
    return text

def main():
    st.title("Text Characteristics Preservation From Images")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    text_extracted = False

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Extract Text"):
            pil_image = Image.open(uploaded_image)
            # Perform OCR on the image
            extracted_text = extract_text(pil_image)
            st.subheader("Extracted Text:")
            st.text(extracted_text)
            # Update the variable to indicate text extraction
            text_extracted = True

    # Display the "Display Sample Image" button only if text has been extracted
    if text_extracted and st.button("Display Sample Image"):
        sample_image_path = "resume.jpg"  # Update with the path to your sample image
        sample_image = Image.open(sample_image_path)

        st.image(sample_image, caption="Sample Image", use_column_width=True)

if __name__ == "__main__":
    main()
