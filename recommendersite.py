# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load data ---
tags_df = pd.read_csv('/ml-latest/genome-tags.csv')  # has 'tag' column
tag_list = tags_df['tag'].tolist()

movies = pd.read_csv('/ml-latest/movies.csv')  # ['movieId', 'title']
movie_tag_matrix = pd.read_csv('/movie_tag_matrix.csv', index_col=0)  # (num_movies, num_tags)

# --- Generate tag embeddings ---
model = SentenceTransformer('all-MiniLM-L6-v2')
tag_vectors = model.encode(tag_list, show_progress_bar=True)  # shape: (num_tags, 384)

# --- Streamlit UI ---

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@700&family=Roboto&family=Roboto+Mono&display=swap');

    /* Page zoom (some browsers may ignore) */
    html {
        zoom: 100%;
    }
    /* Widen Streamlit container */
    .block-container {
        max-width: 50% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Dark gradient background */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
        background: linear-gradient(to bottom right, #111827, #1f2937);
        color: white !important;
    }

    /* Main headings */
    h1, h2, h3, h4, h5, h6,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3 {
        font-family: 'Open Sans', sans-serif !important;
        font-weight: 700 !important;
        color: #a5b4fc !important;
    }

    /* TAGS: Roboto Mono */
    .tag-item {
        font-family: 'Roboto Mono', monospace !important;
        color: white !important;
    }

    /* MOVIE TITLES: Roboto, slightly bigger */
    .movie-title {
        font-family: 'Roboto', sans-serif !important;
        color: white !important;
        font-size: 18px !important;
    }

    /* Inputs and sliders */
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

    /* Paragraphs and inline text */
    p, span, li, div[data-testid="stMarkdownContainer"] * {
        font-family: 'Roboto Mono', monospace !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




st.title("🎬 MoodMatch: Movies by Mood")



user_input = st.text_input("")
if user_input:
    user_vector = model.encode([user_input])  # shape: (1, 384)

    # --- Tag similarity ranking ---
    similarities = cosine_similarity(user_vector, tag_vectors)[0]
    top_n = 20
    top_tag_indices = similarities.argsort()[-top_n:][::-1]
    top_tags = [tag_list[i] for i in top_tag_indices]

    st.subheader("\n \n \n 💫 Matching Moods")
    tag_html = "<p class='tag-item'>" + ", ".join(top_tags) + "</p>"
    st.markdown(tag_html, unsafe_allow_html=True)



    # --- Score movies based on relevant tag weights ---
    movie_scores = movie_tag_matrix.iloc[:, top_tag_indices].values @ similarities[top_tag_indices]
    top_n_movies = st.slider("", 5, 20, 10)
    top_movie_indices = movie_scores.argsort()[-top_n_movies:][::-1]

    st.subheader("\n \n 🎥 Recommended Movies")
    movie_html = ""
    for idx in top_movie_indices:
        title = movies.iloc[idx]['title']
        movie_html += f"<p class='movie-title'> • {title}</p>"
    st.markdown(movie_html, unsafe_allow_html=True)

