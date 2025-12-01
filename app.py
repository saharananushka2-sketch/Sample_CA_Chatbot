import streamlit as st
from groq import Groq
from pypdf import PdfReader
import glob

st.set_page_config(page_title="Credit Analytics LLM Assistant", layout="centered")

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

st.title("📊 Credit Analytics LLM Assistant")
st.write("Ask questions using the power of Llama3 running on Groq.")

api_key = st.text_input("Enter your Groq API Key", type="password")

query = st.text_input("Your credit question:")

if st.button("Ask"):
    if not api_key:
        st.error("Please enter your Groq API key.")
    elif not query.strip():
        st.error("Please enter a question.")
    else:
        client = Groq(api_key=api_key)

        prompt = f"""
        You are a Credit Analytics expert.
        Use ONLY the following internal documents to answer:

        {DOCS}

        QUESTION: {query}
        """

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        answer = response.choices[0].message["content"]
        st.subheader("Answer")
        st.write(answer)
