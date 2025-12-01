import streamlit as st
from pypdf import PdfReader
import glob
import re

st.set_page_config(page_title="Credit Chatbot", layout="centered")

def load_documents():
    text_blocks = []

    for f in glob.glob("documents/*.pdf"):
        try:
            reader = PdfReader(f)
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    # split into paragraphs
                    paragraphs = [p.strip() for p in txt.split("\n\n") if p.strip()]
                    text_blocks.extend(paragraphs)
        except:
            pass

    for f in glob.glob("documents/*.txt"):
        try:
            with open(f, "r", encoding="utf-8") as file:
                paragraphs = [p.strip() for p in file.read().split("\n\n") if p.strip()]
                text_blocks.extend(paragraphs)
        except:
            pass

    return text_blocks

@st.cache_data
def get_docs():
    return load_documents()

DOCS = get_docs()

def find_best_answer(query):
    query = query.lower()
    best_score = 0
    best_para = "Sorry, I couldn’t find relevant information in the documents."

    for para in DOCS:
        words = re.findall(r"\w+", para.lower())
        score = sum(1 for w in words if w in query)

        if score > best_score:
            best_score = score
            best_para = para

    return best_para

# UI
st.title("📊 Credit Analytics Document Assistant")
st.write("Ask questions based on the uploaded PDFs and text files.")

query = st.text_input("Enter your question:")

if st.button("Search"):
    if not query.strip():
        st.write("Please enter a question.")
    else:
        answer = find_best_answer(query)
        st.subheader("Answer")
        st.write(answer)
