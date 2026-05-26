import json
import os
import re

import streamlit as st
from groq import Groq


DATA_FILE = "data.json"
MODEL_NAME = "llama-3.1-8b-instant"


def normalize_docs(raw_docs, default_title="Document"):
    if isinstance(raw_docs, dict):
        raw_docs = raw_docs.get("data", [raw_docs])

    if not isinstance(raw_docs, list):
        raw_docs = [raw_docs]

    docs = []

    for index, item in enumerate(raw_docs, start=1):
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or f"{default_title} {index}")

            if item.get("text"):
                text = str(item["text"])
            else:
                text = ", ".join(
                    f"{key}: {value}" for key, value in item.items() if key != "title"
                )

            if text.strip():
                docs.append({"title": title, "text": text})

        elif isinstance(item, str) and item.strip():
            docs.append({"title": f"{default_title} {index}", "text": item})

    return docs


def load_docs():
    if not os.path.exists(DATA_FILE):
        return []

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return normalize_docs(json.load(f), "Local Document")


def load_uploaded_docs(uploaded_files):
    docs = []
    errors = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")

        if file_name.lower().endswith(".json"):
            try:
                raw_docs = json.loads(content)
                docs.extend(normalize_docs(raw_docs, file_name))
            except json.JSONDecodeError:
                errors.append(f"{file_name} is not valid JSON.")
        elif content.strip():
            docs.append({"title": file_name, "text": content})

    return docs, errors


def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))


def retrieve(query, docs, top_k=3):
    query_words = tokenize(query)
    scored_docs = []

    for doc in docs:
        text = doc.get("text", "")
        score = len(query_words & tokenize(text))
        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


def ask_llm(query, context_docs):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Missing GROQ_API_KEY in your environment."

    client = Groq(api_key=api_key)

    context = "\n\n".join(
        f"Source: {doc.get('title', 'Untitled')}\n{doc.get('text', '')}"
        for doc in context_docs
    )

    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say: "I could not find that in the data."
note: If the data is in table format, use html table format to display the data.
example:
<table>
    <tr>
        <th>Name</th>
        <th>Email</th>
        <th>ID</th>
    </tr>
</table>
Context:
{context}

Question:
{query}
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful RAG assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


st.title("Simple RAG App")

uploaded_files = st.file_uploader(
    "Upload your own .txt or .json files (optional)",
    type=["txt", "json"],
    accept_multiple_files=True,
)

query = st.text_input("Ask a question")

if st.button("Search"):
    uploaded_docs, upload_errors = load_uploaded_docs(uploaded_files) if uploaded_files else ([], [])

    for error in upload_errors:
        st.error(error)

    docs = uploaded_docs or load_docs()

    if not docs:
        st.error("No data found. Upload files or create a data.json file first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        if uploaded_docs:
            st.caption("Using uploaded files")
        else:
            st.caption("Using local data.json")

        results = retrieve(query, docs)

        if not results:
            st.info("No matching documents found.")
        else:
            st.subheader("Retrieved Context")
            for doc in results:
                st.write(f"**{doc.get('title', 'Untitled')}**")
                st.write(doc.get("text", ""))

            answer = ask_llm(query, results)

            st.subheader("Answer")
            st.write(answer)