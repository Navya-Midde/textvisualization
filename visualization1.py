import streamlit as st
import pdfplumber
import docx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import numpy as np
import re

# 📄 Extract text from PDF using pdfplumber
def extract_text_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# 📄 Extract text from DOCX
def extract_text_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# 🧹 Basic text preprocessing without NLTK
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens

# ☁️ Generate Word Cloud
def show_wordcloud(tokens):
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(tokens))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# 📊 Word Frequency Bar Graph
def show_frequency(tokens):
    freq = Counter(tokens).most_common(20)
    words, counts = zip(*freq)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(words), y=list(counts), palette='viridis')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# 🔥 Heatmap of Word Co-occurrence
def show_heatmap(tokens):
    top_words = [word for word, _ in Counter(tokens).most_common(20)]
    matrix = np.zeros((20, 20))
    for i, word1 in enumerate(top_words):
        for j, word2 in enumerate(top_words):
            count = sum(1 for k in range(len(tokens)-1) if tokens[k] == word1 and tokens[k+1] == word2)
            matrix[i][j] = count
    df = pd.DataFrame(matrix, index=top_words, columns=top_words)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap='YlGnBu', annot=True)
    st.pyplot(plt)

# 🚀 Streamlit UI
st.title("📚 Text Visualization App")
uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        raw_text = extract_text_pdf(uploaded_file)
    else:
        raw_text = extract_text_docx(uploaded_file)

    st.subheader("Extracted Text Preview")
    st.write(raw_text[:1000] + "...")  # Show preview

    tokens = preprocess_text(raw_text)

    option = st.selectbox("Choose Visualization", ["Word Cloud", "Word Frequency", "Heatmap"])

    if option == "Word Cloud":
        show_wordcloud(tokens)
    elif option == "Word Frequency":
        show_frequency(tokens)
    elif option == "Heatmap":
        show_heatmap(tokens)
