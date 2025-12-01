import streamlit as st
from groq import Groq
from pypdf import PdfReader
import glob
import numpy as np
import re

st.set_page_config(page_title="Credit Analytics LLM Assistant", layout="centered")

# Load API key from Streamlit Secrets
API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq Client
client = Groq(api_key=API_KEY)

# ----------------------------
# 1. Load Documents & Extract Text
# ----------------------------
def load_documents():
    text = ""
    for f in glob.glob("documents/*.pdf"):
        reader = PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + " "
    for f in glob.glob("documents/*.txt"):
        with open(f, "r", encoding="utf-8") as file:
            text += file.read() + " "
    return text


# ----------------------------
# 2. Chunking
# ----------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# ----------------------------
# 3. Embeddings Function
# ----------------------------
def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)


# ----------------------------
# 4. Vector Store (Embeddings)
# ----------------------------
@st.cache_resource
def prepare_vector_store():
    full_text = load_documents()
    chunks = chunk_text(full_text)
    
    vectors = []
    for ch in chunks:
        emb = embed_text(ch)
        vectors.append(emb)
    
    return chunks, np.array(vectors)

chunks, vectors = prepare_vector_store()


# ----------------------------
# 5. Vector Similarity Search
# ----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_chunk(query):
    query_vector = embed_text(query)
    sims = [cosine_similarity(query_vector, v) for v in vectors]
    best_index = int(np.argmax(sims))
    return chunks[best_index]


# ----------------------------
# 6. LLM Answer Generation
# ----------------------------
def ask_llm(context, query):
    prompt = f"""
You are a Credit Analytics specialist.

Use ONLY the following context from internal documents to answer:

CONTEXT:
{context}

QUESTION:
{query}

Provide a concise expert answer.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message["content"]


# ----------------------------
# 7. Streamlit UI
# ----------------------------
st.title("📊 Credit Analytics LLM Assistant (RAG Powered)")
st.write("Ask any credit question using document-based intelligence.")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if not query.strip():
        st.error("Please enter a question.")
    else:
        best_context = find_best_chunk(query)
        answer = ask_llm(best_context, query)
        
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Most Relevant Context Retrieved")
        st.info(best_context)
