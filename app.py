import streamlit as st
from groq import Groq
from pypdf import PdfReader
import glob

st.set_page_config(page_title="Credit Analytics LLM Assistant", layout="centered")

# Secure: Pull API key from Streamlit secrets
API_KEY = st.secrets["GROQ_API_KEY"]

# Load documents
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

# UI layout
st.title("📊 Credit Analytics LLM Assistant")
st.write("Ask credit questions based on your uploaded documents.")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Print previous chat messages
for role, msg in st.session_state.chat:
    st.markdown(f"**{role}:** {msg}")

query = st.text_input("Your question")

if st.button("Ask"):
    if query.strip():
        st.session_state.chat.append(("You", query))

        client = Groq(api_key=API_KEY)

        prompt = f"""
You are a Credit Analytics expert. 
Use ONLY the following internal documents to answer the user's question.

DOCUMENTS:
{DOCS}

QUESTION:
{query}
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        answer = response.choices[0].message["content"]
        st.session_state.chat.append(("Bot", answer))
        st.experimental_rerun()
