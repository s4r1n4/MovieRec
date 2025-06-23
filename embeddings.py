import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import load_data

# --- Load Datasets ---
movies, tags, scores = load_data()

# --- Create Movie-Tag Matrix (movies as rows, tags as columns) ---
movie_tag_matrix = scores.pivot(index="movieId", columns="tag", values="relevance").fillna(0)
movie_tag_matrix.to_csv("movie_tag_matrix.csv")

# --- Create Tag-Movie Matrix (tags as rows, movies as columns) ---
tag_movie_matrix = scores.pivot(index='tagId', columns='movieId', values='relevance').fillna(0)
tag_matrix_normalized = normalize(tag_movie_matrix, axis=1)

# --- Dimensionality Reduction using TruncatedSVD ---
n_components = 50
svd = TruncatedSVD(n_components=n_components, random_state=42)
tag_embeddings_svd = svd.fit_transform(tag_matrix_normalized)

# --- Save SVD-based Tag Embeddings ---
tag_id_to_name = dict(zip(tags['tagId'], tags['tag']))
tag_names = [tag_id_to_name[tag_id] for tag_id in tag_movie_matrix.index]
tag_embeddings_df = pd.DataFrame(tag_embeddings_svd, index=tag_names)
tag_embeddings_df.columns = [f"dim_{i+1}" for i in range(n_components)]
tag_embeddings_df.to_csv("tag_embeddings.csv")

# --- SentenceTransformer Embeddings ---
tag_list = tags['tag'].tolist()
model = SentenceTransformer('all-MiniLM-L6-v2')
tag_embeddings_st = model.encode(tag_list, show_progress_bar=True)

# --- Get User Input & Encode ---
user_input = input("Describe your mood or preferences: ")
user_vector = model.encode([user_input])[0]

# --- Compute Similarity between input and all tags ---
similarities = cosine_similarity([user_vector], tag_embeddings_st)[0]
tag_similarities = pd.DataFrame({'tag': tag_list, 'similarity': similarities})

# --- Top N Relevant Tags ---
top_n = 30
top_tags = tag_similarities.sort_values(by='similarity', ascending=False).head(top_n)
print("Top Tags:", top_tags['tag'].tolist())

# --- Filter scores for top tagIds ---
top_tag_ids = tags[tags['tag'].isin(top_tags['tag'])]['tagId'].tolist()
relevant_scores = scores[scores['tagId'].isin(top_tag_ids)]

# --- Aggregate score per movie ---
movie_scores = relevant_scores.groupby('movieId')['relevance'].mean().reset_index()

# --- Merge with movie titles ---
top_movies = movie_scores.merge(movies, on='movieId').sort_values('relevance', ascending=False)

# --- Display Top Movies ---
print("\n🎬 Top Movie Recommendations:")
for i, row in top_movies.head(15).iterrows():
    print(f"{row['title']} (score: {row['relevance']:.3f})")
