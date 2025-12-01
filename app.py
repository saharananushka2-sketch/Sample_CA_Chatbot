import streamlit as st
from transformers import pipeline
from pypdf import PdfReader
import glob

st.set_page_config(page_title="Credit Analytics Chatbot", layout="centered")

# Load fast model
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

model = load_model()

# Load documents
def load_docs():
    text = ""
    for file in glob.glob("documents/*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            text += f.read() + "\n"

    for file in glob.glob("documents/*.pdf"):
        reader = PdfReader(file)
        for page in reader.pages:
            t = page.extract_text() or ""
            text += t + "\n"
    return text

DOCS = load_docs()

st.title("📊 Credit Analytics Chatbot")
st.caption("Ask questions based on your uploaded documents.")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat bubbles
for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")

# User input
user_input = st.text_input("Ask something...")

if st.button("Send"):
    if user_input.strip():
        st.session_state.history.append(("user", user_input))

        prompt = f"Use ONLY this document:\n{DOCS}\n\nUser: {user_input}"
        response = model(prompt, max_length=200)[0]["generated_text"]

        st.session_state.history.append(("assistant", response))
        st.experimental_rerun()
