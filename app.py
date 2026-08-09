# 🧠 PubMed Medical Summarizer (Streamlit App)
# Description: Biomedical abstract summarization using a fine-tuned T5-small model (PubMed dataset)

import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import requests
import zipfile
import time

device = "GPU" if torch.cuda.is_available() else "CPU"

# -----------------------------
# 🔧 CONFIGURATION
# -----------------------------
st.set_page_config(page_title="🧠 PubMed Medical Summarizer", layout="centered")

MODEL_DIR = "t5_pubmed_model"
ZIP_NAME = "t5_pubmed_model.zip"


URL = "https://github.com/sha-md/t5-pubmed-summarizer/releases/download/v1.0-pubmed/t5_pubmed_model_zip.zip"

# -----------------------------
# 🎨 HERO HEADER
# -----------------------------
st.markdown("""
<div style="
    background: linear-gradient(90deg,#0f4c81,#2563eb);
    padding:30px;
    border-radius:15px;
    color:white;
    text-align:center;
">

<h1>🧠 PubMed Medical Summarizer</h1>

<p style="font-size:18px;">
Generate concise biomedical summaries using a fine-tuned
<b>T5-small Transformer</b> trained on the
<b>PubMed Summarization Dataset</b>.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    label="🤖 Model",
    value="T5-small"
)

c2.metric(
    label="📚 Dataset",
    value="PubMed"
)

c3.metric(
    label="🧬 Task",
    value="Summarization"
)

c4.metric(
    label="⚡ Parameters",
    value="60M"
)
st.divider()

# -----------------------------
# 🧩 DOWNLOAD & EXTRACT MODEL
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_from_github():
    """Download fine-tuned T5 model from GitHub Releases (if not already present)."""
    if not os.path.exists(MODEL_DIR):
        st.info("Downloading pretrained model (first run only)...")
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
                st.toast("Model downloaded successfully.")
            else:
                st.error("❌ The downloaded file is not a valid ZIP. Please check your GitHub release link.")
                st.stop()
        else:
            st.error(f"❌ Download failed (status code: {response.status_code}).")
            st.stop()
    else:
        pass

    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tokenizer, model


# -----------------------------
# 🧠 LOAD MODEL
# -----------------------------
with st.spinner("Loading T5 model..."):
    tokenizer, model = load_model_from_github()

st.toast("✅ AI model ready")


# -----------------------------
# 🩺 SUMMARIZATION FUNCTION
# -----------------------------
def summarize_text(text, summary_length, beam_size):

    input_text = "summarize: " + text
    inputs = inputs.to(model.device)
    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    if summary_length == "Short":
        max_len = 60
        min_len = 20

    elif summary_length == "Medium":
        max_len = 120
        min_len = 35

    else:
        max_len = 180
        min_len = 60

    summary_ids = model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        num_beams=beam_size,
        repetition_penalty=2.5,
        length_penalty=1.2,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# -----------------------------
# 📘 SIDEBAR INFO
# -----------------------------
st.sidebar.title("📘 Project Information")

st.sidebar.markdown("""
### 🤖 Model

**T5-small**

Fine-tuned for biomedical abstractive summarization.

---

### 📚 Dataset

**ccdv/pubmed-summarization**

Derived from the PubMed Open Access Subset.

---

### 🎯 Task

Biomedical Research Abstract Summarization

---

### ⚙️ Framework

- Hugging Face Transformers
- PyTorch
- Streamlit

---
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "💡 Paste a biomedical abstract or load one of the sample examples to generate an AI-powered summary."
)


# -----------------------------
# 🧾 MAIN INTERFACE
# -----------------------------

st.markdown("## 🧬 Summarize Biomedical Text")

st.write(
    """
Paste a biomedical abstract, clinical study, or research paragraph below.
The fine-tuned T5 model will generate a concise abstractive summary while
preserving the key medical information.
"""
)

with st.expander("💡 Load an Example Abstract"):

    if st.button("Child Malnutrition Study"):
        st.session_state["sample_text"] = """
A recent systematic analysis showed that in 2011,
314 million children younger than five years were mildly,
moderately or severely stunted.
The prevalence of malnutrition among Iranian school children
ranged from 6% to 16%.
Anthropometric data from Tehran showed similar findings.
"""

    if st.button("Cardiovascular Study"):
        st.session_state["sample_text"] = """
Elevated cholesterol is one of the major causes of cardiovascular
disease. Clinical trials demonstrated that statins reduced LDL
cholesterol while lowering cardiovascular risk.
"""

    if st.button("COVID Vaccine Study"):
        st.session_state["sample_text"] = """
Clinical trials showed that mRNA COVID-19 vaccines achieved over
90% efficacy against symptomatic infection while maintaining an
acceptable safety profile.
"""

sample_text = st.text_area(
    "Biomedical Text",
    value=st.session_state.get("sample_text",""),
    height=280,
    placeholder="Paste biomedical abstract here..."
)

col1, col2 = st.columns(2)

summary_length = col1.selectbox(
    "Summary Length",
    ["Short","Medium","Detailed"],
    index=1
)

beam_size = col2.selectbox(
    "Generation Quality",
    [2,4,6],
    index=2
)
# -----------------------------
# 🚀 SUMMARIZE BUTTON
# -----------------------------
if st.button("🔍 Summarize"):
    if not sample_text.strip():
        st.warning("⚠️ Please enter or load text before summarizing.")
    else:
        with st.spinner("Generating summary..."):
            
            start = time.time()
            summary = summarize_text(
                sample_text,
                summary_length,
                beam_size
            )
            end = time.time()

            inference_time = end - start
            st.session_state["inference_time"] = inference_time
        st.success("✅ Summary generated successfully!")
        st.caption(
            f"⚡ Generated in {inference_time:.2f} seconds"
        )

        original_words = len(sample_text.split())
        summary_words = len(summary.split())
        
        compression = (
            (1 - summary_words / original_words) * 100
            if original_words > 0
            else 0
        )

        reading_time = max(1, round(summary_words / 200))
        
        st.markdown("## 📊 Summary Statistics")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        
        c1.metric(
            "Original Words",
            original_words
        )

        c2.metric(
            "Summary Words",
            summary_words
        )
        
        c3.metric(
            "Compression",
            f"{compression:.1f}%"
        )
        
        c4.metric(
            "Reading Time",
            f"{reading_time} min"
        )

        c5.metric(
            "Input Characters",
            len(sample_text)
        )


        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader("📄 Original Abstract")
        
            with st.container(border=True):
                st.markdown(sample_text)

        with right:

            st.subheader("🧠 AI Generated Summary")
        
            with st.container(border=True):
                st.write(summary)

        st.download_button(
            "📥 Download Summary",
            summary,
            file_name="medical_summary.txt",
            mime="text/plain"
        )

inference_time = st.session_state.get(
    "inference_time",
    None
)

with st.expander("🔍 Model Details"):

    st.markdown(f"""
**Model**

- T5-small
- Fine-tuned on PubMed
- Device:
    {device}
- Parameters:
    ~60 Million

**Generation Settings**

- Beam Size: {beam_size}

- Summary Length: {summary_length}

- Maximum Tokens:
    {"60" if summary_length=="Short" else "120" if summary_length=="Medium" else "180"}


- Inference Time:
    {f"{inference_time:.2f} sec" if inference_time else "Not generated yet"}
""")

with st.expander("⚠️ Model Limitations"):

    st.markdown("""
This summarizer is intended for educational and research purposes.

Limitations include:

- Long documents are truncated to the first 512 tokens.
- Generated summaries may omit fine-grained details.
- Outputs should not be considered medical advice.
- Human verification is recommended for clinical or research decisions.
""")

st.divider()

st.caption(
    "Built with Streamlit • Hugging Face Transformers • PyTorch • Fine-tuned T5-small"
)

