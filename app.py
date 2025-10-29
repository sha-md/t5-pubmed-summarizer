import streamlit as st
import requests
import zipfile
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------------------------------------------------------
# 📦 CONFIGURATION
# -------------------------------------------------------------------
MODEL_DIR = "t5_pubmed_model"
ZIP_NAME = "t5_pubmed_model.zip"
FILE_ID = "1XZF_zbpWr-JIhlk1KkFR9gO4H9UrOiLv"  
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# -------------------------------------------------------------------
# ⚙️ FUNCTION: Download & Load Model
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_from_drive():
    if not os.path.exists(MODEL_DIR):
        st.info("📦 Downloading model (~500 MB)... Please wait 3–5 minutes.")

        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(URL, headers=headers, stream=True) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                st.error("❌ Google Drive returned an HTML page. Make sure file is shared with 'Anyone with the link'.")
                st.stop()

            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0
            with open(ZIP_NAME, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        percent = (downloaded_size / total_size) * 100 if total_size else 0
                        st.write(f"Downloading... {percent:.1f}%")

        st.info("📂 Extracting model files...")
        with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(ZIP_NAME)
        st.success("✅ Model downloaded and extracted successfully!")

    # Load model
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    return tokenizer, model


# -------------------------------------------------------------------
# 🧠 STREAMLIT APP UI
# -------------------------------------------------------------------
st.title("🧠 PubMed Medical Summarizer")
st.write("Fine-tuned **T5-small** model for biomedical abstract summarization.")

st.markdown("### 📘 About this App")
st.markdown(
    "This tool summarizes biomedical research abstracts using a fine-tuned Transformer model "
    "trained on the PubMed dataset. Enter a long abstract below and get a concise summary instantly!"
)

# Load model
tokenizer, model = load_model_from_drive()

# -------------------------------------------------------------------
# 🧾 INPUT AREA
# -------------------------------------------------------------------
st.subheader("🩺 Enter a medical abstract:")
article = st.text_area("Paste the abstract text here:", height=250)

if st.button("✨ Summarize"):
    if not article.strip():
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Generating summary... ⏳"):
            inputs = tokenizer("summarize: " + article, return_tensors="pt", truncation=True, max_length=512)
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=40,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.success("✅ Summary Generated!")
        st.markdown("### 🩸 **Generated Summary:**")
        st.write(summary)

# -------------------------------------------------------------------
# 📚 FOOTER
# -------------------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Transformers, and PyTorch.")
