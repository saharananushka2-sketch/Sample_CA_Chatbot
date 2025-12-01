import streamlit as st
from transformers import pipeline
from pypdf import PdfReader
import glob

st.set_page_config(page_title="Credit Chatbot", layout="centered")

@st.cache_resource
def load_model():
    return pipeline("question-answering", model="deepset/tinyroberta-squad2")

model = load_model()

def load_docs():
    text = ""
    for f in glob.glob("documents/*.pdf"):
        try:
            reader = PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t
        except:
            pass

    for f in glob.glob("documents/*.txt"):
        try:
            with open(f, "r", encoding="utf-8") as file:
                text += file.read()
        except:
            pass
    return text

DOCS = load_docs()

st.title("📊 Credit Analytics Chatbot")
st.write("Ask questions based only on the uploaded documents.")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.markdown(f"**{role}:** {msg}")

query = st.text_input("Your question")

if st.button("Ask"):
    if query.strip():
        st.session_state.chat.append(("You", query))

        answer = model(question=query, context=DOCS)
        response = answer["answer"]

        st.session_state.chat.append(("Bot", response))
        st.experimental_rerun()
