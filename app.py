# app.py
# 🧠 PubMed Summarizer — Streamlit App
# Author: [Your Name]
# Description: Transformer-based medical text summarizer fine-tuned on PubMed dataset (T5-small)

import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ----------------------------------------------
# ⚙️ Page Configuration
# ----------------------------------------------
st.set_page_config(
    page_title="🧠 PubMed Summarizer",
    page_icon="🩺",
    layout="wide",
)

# ----------------------------------------------
# 🚀 Load Model and Tokenizer
# ----------------------------------------------
@st.cache_resource
def load_model():
    model_path = "./pubmed_t5_model"  # Folder containing your saved model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# ----------------------------------------------
# 🏷️ App Title & Description
# ----------------------------------------------
st.title("🧠 PubMed Summarizer")
st.markdown("""
This app uses a fine-tuned **T5 Transformer** to generate concise, factual summaries of **biomedical research abstracts**.  
Model trained on the **PubMed Summarization Dataset** using Hugging Face 🤗 and PyTorch 🧩.

---
""")

# ----------------------------------------------
# ✍️ User Input
# ----------------------------------------------
st.subheader("📘 Enter a Medical Abstract")
text_input = st.text_area(
    "Paste your abstract or article text below 👇",
    height=250,
    placeholder="Example: A recent systematic analysis showed that in 2011, 314 million children under 5 years were mildly, moderately, or severely stunted..."
)

# ----------------------------------------------
# 🔍 Generate Summary
# ----------------------------------------------
if st.button("✨ Summarize"):
    if text_input.strip():
        with st.spinner("🩺 Generating summary... please wait"):
            inputs = tokenizer(
                "summarize: " + text_input,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=40,
                num_beams=4,
                early_stopping=True
            )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display Results
        st.subheader("🩺 Generated Summary:")
        st.success(summary)
    else:
        st.warning("⚠️ Please enter text before clicking Summarize.")

# ----------------------------------------------
# 🧾 Sidebar Information
# ----------------------------------------------
st.sidebar.header("ℹ️ About this App")
st.sidebar.markdown("""
**PubMed Summarizer**  
Built with **T5-small**, fine-tuned on biomedical abstracts.

**Tech Stack:**  
- 🧠 Transformers (Hugging Face)  
- 🔥 PyTorch  
- 🧩 Streamlit  

**Use Cases:**  
- Summarizing PubMed abstracts  
- Generating medical research briefs  
- Helping healthcare professionals quickly review studies  
""")

st.sidebar.write("---")
st.sidebar.info("📄 *Model: T5-small (fine-tuned)*")

# ----------------------------------------------
