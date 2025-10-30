# 🧠 PubMed Medical Summarizer (Streamlit App)
# Description: Summarizes biomedical research abstracts using a fine-tuned T5-small model trained on the PubMed dataset.

import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import requests
import zipfile

# -----------------------------
# 🔧 CONFIGURATION
# -----------------------------
st.set_page_config(page_title="🧠 PubMed Medical Summarizer", layout="centered")

MODEL_DIR = "t5_pubmed_model"
ZIP_NAME = "t5_pubmed_model.zip"

# ✅ Use your GitHub Release direct link
URL = "https://github.com/sha-md/t5-pubmed-summarizer/releases/download/v1.0-pubmed/t5_pubmed_model_zip.zip"


# -----------------------------
# 🧩 HELPER: DOWNLOAD & EXTRACT MODEL
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_from_github():
    """Download the fine-tuned T5 model from GitHub Releases if not already present."""
    if not os.path.exists(MODEL_DIR):
        st.info("📦 Downloading model (~500 MB)... Please wait 3–5 minutes.")
        response = requests.get(URL, stream=True)
        if response.status_code == 200:
            with open(ZIP_NAME, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            # ✅ Extract safely
            if zipfile.is_zipfile(ZIP_NAME):
                with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
                    zip_ref.extractall(".")
                os.remove(ZIP_NAME)
                st.success("✅ Model downloaded and extracted successfully!")
            else:
                st.error("❌ The downloaded file is not a valid ZIP. Please check your GitHub release link.")
                st.stop()
        else:
            st.error(f"❌ Download failed with status code: {response.status_code}")
            st.stop()
    else:
        st.info("✅ Model found locally (skipping download).")

    # 🔹 Load the model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    return tokenizer, model


# -----------------------------
# 🧠 LOAD MODEL
# -----------------------------
st.title("🧠 PubMed Medical Summarizer")
st.caption("Fine-tuned T5-small model for biomedical abstract summarization.")

with st.spinner("Loading model..."):
    tokenizer, model = load_model_from_github()
st.success("Model ready for summarization!")


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
# 📋 USER INTERFACE
# -----------------------------
st.subheader("📘 About this App")
st.markdown("""
This tool summarizes **biomedical research abstracts** using a fine-tuned Transformer model trained on the **PubMed dataset**.  
Enter a long abstract below and get a concise, research-style summary instantly!
""")

sample_text = st.text_area(
    "Paste your abstract or research paragraph here 👇",
    height=250,
    placeholder="Enter biomedical text here..."
)

if st.button("🔍 Summarize"):
    if not sample_text.strip():
        st.warning("⚠️ Please enter some text before summarizing.")
    else:
        with st.spinner("Generating summary... ⏳"):
            summary = summarize_text(sample_text)
        st.success("✅ Summary generated successfully!")
        st.markdown("### 🩺 Generated Summary:")
        st.write(summary)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, PyTorch, and Hugging Face Transformers.")
