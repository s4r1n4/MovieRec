import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Load CSVs
movies = pd.read_csv("/ml-latest/movies.csv")
tags = pd.read_csv("/ml-latest/genome-tags.csv")
scores = pd.read_csv("/ml-latest/genome-scores.csv")

# Merge tag names into genome scores
scores = scores.merge(tags, on="tagId")

# Pivot to get a tag vector per movie
tag_matrix = scores.pivot(index="movieId", columns="tag", values="relevance").fillna(0)

# Step 2: Create Tag-Movie Matrix (tags as rows, movies as columns)
# Pivot the table to create a tag-movie matrix with scores as values
tag_movie_matrix = scores.pivot(index='tagId', columns='movieId', values='relevance').fillna(0)

#Normalize the matrix so that each tag is a unit vector
tag_matrix_normalized = normalize(tag_movie_matrix, axis=1)

# Step 3: Dimensionality Reduction using TruncatedSVD (similar to PCA but for sparse/high-dim data)
n_components = 50  # You can tune this number depending on how many dimensions you want
svd = TruncatedSVD(n_components=n_components, random_state=42)
tag_embeddings = svd.fit_transform(tag_matrix_normalized)

# Step 4: Put the embeddings into a DataFrame with tag names
tag_id_to_name = dict(zip(tags['tagId'], tags['tag']))
tag_names = [tag_id_to_name[tag_id] for tag_id in tag_movie_matrix.index]
tag_embeddings_df = pd.DataFrame(tag_embeddings, index=tag_names)
tag_embeddings_df.columns = [f"dim_{i+1}" for i in range(n_components)]

# View the embeddings
print(tag_embeddings_df.head())

#Save to CSV
tag_embeddings_df.to_csv("tag_embeddings.csv")


#Find similar tags
def find_similar_tags(tag, top_n=10):
    if tag not in tag_embeddings_df.index:
        print(f"Tag '{tag}' not found.")
        return
    vec = tag_embeddings_df.loc[[tag]]
    sims = cosine_similarity(vec, tag_embeddings_df)[0]
    top_indices = sims.argsort()[::-1][1:top_n+1]
    return tag_embeddings_df.index[top_indices]

# Example:
print(find_similar_tags("romance"))
