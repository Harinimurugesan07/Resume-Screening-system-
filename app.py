import streamlit as st
import joblib
import PyPDF2
import docx2txt
import os

# Load the trained model and TF-IDF vectorizer
model = joblib.load('resume_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.name.endswith('.docx'):
        return docx2txt.process(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        return str(uploaded_file.read(), 'utf-8')
    else:
        return None

# Streamlit UI
st.title("Resume Screening System")
st.markdown("Upload a resume in .pdf, .docx, or .txt format")

uploaded_file = st.file_uploader("Choose a resume file", type=['pdf', 'docx', 'txt'])

if uploaded_file is not None:
    resume_text = extract_text_from_file(uploaded_file)
    
    if resume_text:
        st.subheader("Extracted Resume Text:")
        st.write(resume_text)

        # Vectorize and predict
        resume_vector = tfidf.transform([resume_text])
        prediction = model.predict(resume_vector)[0]

        st.success(f"Predicted Category: *{prediction}*")
    else:
        st.error("Unsupported file format or error reading file.")