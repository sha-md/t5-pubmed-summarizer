# 🧠 PubMed Medical Summarizer (Streamlit App)
# Description: Biomedical abstract summarization using a fine-tuned T5-small model (PubMed dataset)

import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import requests
import zipfile
import time

# -----------------------------
# 🔧 CONFIGURATION
# -----------------------------
st.set_page_config(page_title="🧠 PubMed Medical Summarizer", layout="centered")

MODEL_DIR = "t5_pubmed_model"
ZIP_NAME = "t5_pubmed_model.zip"


URL = "https://github.com/sha-md/t5-pubmed-summarizer/releases/download/v1.0-pubmed/t5_pubmed_model_zip.zip"

# -----------------------------
# 🎨 HEADER SECTION
# -----------------------------
st.markdown("""
<div style='text-align: center; padding: 25px; border-radius: 12px; background: #f0f8ff;'>
    <h1 style='color:#1f77b4;'>🧠 PubMed Medical Summarizer</h1>
    <p style='font-size:17px; color:#333;'>An AI tool that condenses complex biomedical abstracts into concise, research-ready summaries.</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# 🧩 DOWNLOAD & EXTRACT MODEL
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_from_github():
    """Download fine-tuned T5 model from GitHub Releases (if not already present)."""
    if not os.path.exists(MODEL_DIR):
        st.info("📦 Downloading model (~500 MB)... Please wait 3–5 minutes.")
        response = requests.get(URL, stream=True)
        if response.status_code == 200:
            with open(ZIP_NAME, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if zipfile.is_zipfile(ZIP_NAME):
                with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
                    zip_ref.extractall(".")
                os.remove(ZIP_NAME)
                st.success("✅ Model downloaded and extracted successfully!")
            else:
                st.error("❌ The downloaded file is not a valid ZIP. Please check your GitHub release link.")
                st.stop()
        else:
            st.error(f"❌ Download failed (status code: {response.status_code}).")
            st.stop()
    else:
        st.info("✅ Model found locally — skipping download.")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    return tokenizer, model


# -----------------------------
# 🧠 LOAD MODEL
# -----------------------------
with st.spinner("Loading AI model... ⏳"):
    tokenizer, model = load_model_from_github()
st.success("✅ Model loaded and ready for summarization!")


# -----------------------------
# 🩺 SUMMARIZATION FUNCTION
# -----------------------------
def summarize_text(text):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# -----------------------------
# 📘 SIDEBAR INFO
# -----------------------------
st.sidebar.title("📘 About This App")
st.sidebar.markdown("""
**PubMed Summarizer** condenses biomedical research abstracts into short summaries  
using a fine-tuned [T5-small](https://huggingface.co/t5-small) model.

🧩 **Built With**
- PyTorch  
- Hugging Face Transformers  
- Streamlit  

📚 **Dataset**
[ccdv/pubmed-summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization)

👩‍💻 **Developer**
Shabnam   
 Built for research and learning.
""")


# -----------------------------
# 🧾 MAIN INTERFACE
# -----------------------------
st.markdown("### 🧬 Try the Summarizer")

# 💡 Demo sample option
with st.expander("💡 Need an example? Click to load a sample abstract"):
    if st.button("Use Example Text"):
        st.session_state["sample_text"] = """A recent systematic analysis showed that in 2011,
        314 million children younger than 5 years were mildly, moderately, or severely stunted.
        The prevalence of malnutrition among Iranian school children ranged from 6% to 16%.
        Anthropometric data from Tehran showed similar findings."""
        st.success("✅ Sample abstract loaded below!")

# Text input
sample_text = st.text_area(
    "📄 Paste your abstract or research paragraph here:",
    value=st.session_state.get("sample_text", ""),
    height=250,
    placeholder="Enter biomedical text here..."
)

# -----------------------------
# 🚀 SUMMARIZE BUTTON
# -----------------------------
if st.button("🔍 Summarize"):
    if not sample_text.strip():
        st.warning("⚠️ Please enter or load text before summarizing.")
    else:
        with st.spinner("Analyzing biomedical content..."):
            progress = st.progress(0)
            for percent in range(0, 101, 20):
                time.sleep(0.2)
                progress.progress(percent)
            summary = summarize_text(sample_text)
        st.success("✅ Summary generated successfully!")

        st.markdown("### 🧾 Original Text:")
        st.write(sample_text[:600] + "..." if len(sample_text) > 600 else sample_text)

        st.markdown("### ✨ Generated Summary:")
        st.info(summary)


# -----------------------------

