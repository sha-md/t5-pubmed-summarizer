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
    """Reliable Google Drive file downloader with confirmation token handling."""
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None

    # Look for download confirmation token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    # Check for HTML (Google warning page)
    if "text/html" in response.headers.get("Content-Type", ""):
        raise ValueError(
            "❌ Google Drive returned an HTML page instead of a ZIP. "
            "Please ensure the file is shared with 'Anyone with the link'."
        )

    total = int(response.headers.get("Content-Length", 0))
    bytes_written = 0
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
                if progress_callback and total:
                    pct = int(bytes_written / total * 100)
                    progress_callback(pct)

    return destination


@st.cache_resource
def load_model_from_drive(file_id):
    """Downloads, extracts, and loads the model once."""
    ZIP_NAME = "t5_pubmed_model.zip"
    MODEL_DIR = "t5_pubmed_model"

    if not os.path.isdir(MODEL_DIR):
        st.warning("📦 Downloading model (~500MB)... Please wait 3–5 minutes.")
        progress = st.progress(0)

        def update_progress(pct):
            progress.progress(pct)

        # Download ZIP
        try:
            download_file_from_google_drive(file_id, ZIP_NAME, progress_callback=update_progress)
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            st.stop()

        # Verify it’s a valid ZIP
        if not zipfile.is_zipfile(ZIP_NAME):
            st.error("❌ The downloaded file is not a valid ZIP. Check your Google Drive share settings.")
            st.stop()

        # Extract ZIP
        with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
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

