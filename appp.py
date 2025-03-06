import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text


def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_sim = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_sim

# Streamlit UI Enhancements
st.set_page_config(page_title="Billa.AI", layout="wide")

# Adding 'Billa.AI' text in red and yellow
st.markdown("""
    <div style="text-align: center; font-size: 20px; font-weight: bold; color: red; background-color: yellow; padding: 5px; border-radius: 10px; width: fit-content; margin: auto;">
        Billa.AI
    </div>
""", unsafe_allow_html=True)

# Display a cat and a snake image
cat_image = "https://cdn2.thecatapi.com/images/MTY3ODIyMQ.jpg"
snake_image = "https://upload.wikimedia.org/wikipedia/commons/3/3b/Reptile_Snake.jpg"

st.markdown("""
    <div style="text-align: center; margin-top: 10px;">
""", unsafe_allow_html=True)

st.image(cat_image, width=200, caption="Cute Cat")
st.image(snake_image, width=200, caption="Snake")

st.markdown("""
    </div>
""", unsafe_allow_html=True)

st.markdown("""<h1 style='text-align: center; color: #4A90E2;'>AI Resume Ranking System</h1>""", unsafe_allow_html=True)
st.markdown("""<p style='text-align: center;'>Upload resumes and enter the job description to find the best match.</p>""", unsafe_allow_html=True)

st.sidebar.header("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

st.sidebar.header("Job Description")
job_description = st.sidebar.text_area("Enter Job Description", height=150)

if uploaded_files and job_description:
    st.subheader("Ranked Resumes")
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    scores = rank_resumes(job_description, resumes)
    
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    # Apply custom styling
    st.dataframe(results.style.format({"Score": "{:.2f}"}).bar(subset=["Score"], color='#4A90E2'))
    
    st.success("Ranking Completed! The top-matching resumes are displayed.")

else:
    st.warning("Please upload resumes and enter a job description to proceed.")
