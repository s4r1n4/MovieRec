# 🎬 Movie Recommender by Mood

This is a Streamlit-based web app that recommends movies based on the mood or feeling you describe in plain English. Using semantic embeddings of movie genome tags and user input, it generates a personalized list of film suggestions.

![screenshot](path/to/your/screenshot.png)

---

## 💡 Features

- **Natural language input** for describing your mood (e.g., "tragic but beautiful", "funny and romantic").
- Uses **Sentence Transformers (MiniLM)** to match your mood with relevant movie tags.
- Recommends top **matching movies** from the MovieLens dataset.
- Beautiful dark-themed UI with custom fonts and layout.
- Tag clusters and semantic outliers for better tag organization (optional).

---

## 🧠 How it works

1. Each movie tag is embedded using `all-MiniLM-L6-v2` from the Sentence Transformers library.
2. User input is embedded the same way and matched against tag embeddings via cosine similarity.
3. The most similar tags are used to score and rank movies based on genome relevance.
4. Results are shown in a clean UI via Streamlit.

---

## 📁 Project Structure

