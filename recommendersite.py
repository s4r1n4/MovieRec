import streamlit as st
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_data

# --- Load data ---
movies, tags_df, scores = load_data()
tag_list = tags_df['tag'].tolist()

# Load precomputed movie-tag matrix
movie_tag_matrix = pd.read_csv('data/movie_tag_matrix.csv', index_col=0)

# --- Generate tag embeddings ---
model = SentenceTransformer('all-MiniLM-L6-v2')
tag_vectors = model.encode(tag_list, show_progress_bar=True)

# --- Custom Styles ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@700&family=Roboto&family=Roboto+Mono&display=swap');

    html {
        zoom: 100%;
    }

    .block-container {
        max-width: 70% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
        background: linear-gradient(to bottom right, #111827, #1f2937);
        color: white !important;
    }

    h1, h2, h3, h4, h5, h6,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3 {
        font-family: 'Open Sans', sans-serif !important;
        font-weight: 700 !important;
        color: #a5b4fc !important;
        font-size: 1.6rem !important;
    }

    .tag-item {
        font-family: 'Roboto Mono', monospace !important;
        color: white !important;
        font-size: 1rem;
    }

    .movie-title {
        font-family: 'Roboto', sans-serif !important;
        color: white !important;
        font-size: 1.1rem !important;
    }

    input[type="text"], textarea, .stTextInput input {
        color: white !important;
        background-color: #374151 !important;
        border: 1px solid #4b5563 !important;
        font-family: 'Roboto Mono', monospace !important;
    }

    .stSlider > div, .stSlider label {
        font-family: 'Roboto Mono', monospace !important;
        color: white !important;
    }

    p, span, li, div[data-testid="stMarkdownContainer"] * {
        font-family: 'Roboto Mono', monospace !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Streamlit App ---
st.title("🎬 MoodMatch: Movies by Mood")

user_input = st.text_input("")

if user_input:
    user_vector = model.encode([user_input])
    similarities = cosine_similarity(user_vector, tag_vectors)[0]

    top_n = 20
    top_tag_indices = similarities.argsort()[-top_n:][::-1]
    top_tags = [tag_list[i] for i in top_tag_indices]

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.subheader("💫 Matching Moods")
    tag_html = "<p class='tag-item'>" + ", ".join(top_tags) + "</p>"
    st.markdown(tag_html, unsafe_allow_html=True)

    movie_scores = movie_tag_matrix.iloc[:, top_tag_indices].values @ similarities[top_tag_indices]
    top_n_movies = st.slider("", 5, 20, 10)
    top_movie_indices = movie_scores.argsort()[-top_n_movies:][::-1]

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎥 Recommended Movies")
    movie_html = ""
    for idx in top_movie_indices:
        title = movies.loc[movies.index == idx, 'title'].values[0]
        movie_html += f"<p class='movie-title'> • {title}</p>"
    st.markdown(movie_html, unsafe_allow_html=True)
