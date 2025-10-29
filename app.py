import os
import zipfile
import requests
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ==============================================
# CONFIGURATION
# ==============================================
DRIVE_FILE_ID = "1Nt7XvedS7h643zkEBwc_Fi3wN4tf3Yn2"  
ZIP_NAME = "t5_pubmed_model_zip.zip"      
MODEL_DIR = "t5_pubmed_model"         

st.set_page_config(page_title="🧠 PubMed Summarizer", page_icon="🩺", layout="wide")

# ==============================================
# UTILITY FUNCTIONS
# ==============================================
def download_file_from_google_drive(file_id, destination, progress_callback=None):
    """Downloads large file from Google Drive, handling the confirmation token for large files."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning') or key.startswith('download'):
            token = value
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    total = response.headers.get('Content-Length')
    total = int(total) if total is not None else None
    CHUNK_SIZE = 32768
    bytes_written = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
                if progress_callback and total:
                    progress_callback(bytes_written, total)
    return destination


@st.cache_resource
def load_model_from_drive(file_id=DRIVE_FILE_ID):
    """Downloads model zip from Google Drive, extracts, and loads the tokenizer and model."""
    if not os.path.isdir(MODEL_DIR):
        st.warning("📦 Downloading model from Google Drive (~500MB)...")
        with st.spinner("Downloading model... Please wait 3–5 minutes (only once)."):
            progress_bar = st.progress(0)

            def progress_callback(written, total):
                pct = int(written / total * 100)
                progress_bar.progress(pct)

            download_file_from_google_drive(file_id, ZIP_NAME, progress_callback=progress_callback)
            with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
                zip_ref.extractall(MODEL_DIR)
            os.remove(ZIP_NAME)
            st.success("✅ Model downloaded and extracted successfully!")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    return tokenizer, model


# ==============================================
# MAIN APP UI
# ==============================================
st.title("🧠 PubMed Medical Summarizer")
st.caption("Fine-tuned **T5-small** model for biomedical abstract summarization.")

with st.expander("📘 About this App"):
    st.markdown("""
    This app uses a fine-tuned **T5-small Transformer** trained on the 
    [PubMed Summarization Dataset](https://huggingface.co/datasets/ccdv/pubmed-summarization).  
    It converts **long biomedical research abstracts** into concise summaries.
    """)

# Load model
tokenizer, model = load_model_from_drive()

# ==============================================
# USER INPUT
# ==============================================
st.subheader("🩺 Enter your medical abstract below:")
article_text = st.text_area("Paste the research abstract or text you want to summarize:", height=200)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    max_length = st.slider("Maximum Summary Length (tokens)", 50, 300, 150)
with col2:
    min_length = st.slider("Minimum Summary Length (tokens)", 20, 100, 40)
with col3:
    num_beams = st.slider("Beam Search (Quality)", 2, 6, 4)

generate_button = st.button("✨ Generate Summary")

# ==============================================
# GENERATION
# ==============================================
if generate_button:
    if not article_text.strip():
        st.error("⚠️ Please enter a text to summarize.")
    else:
        st.info("🧠 Generating summary... (this may take a few seconds)")
        inputs = tokenizer("summarize: " + article_text, return_tensors="pt", truncation=True, max_length=512)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.success("✅ Summary Generated:")
        st.write(summary)

        with st.expander("📄 View Original Text"):
            st.write(article_text)

# ==============================================

