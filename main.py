from utils import load_data
from embeddings import compute_tag_embeddings_svd, compute_tag_embeddings_semantic
from clustering import cluster_tags, plot_tsne, find_outliers
from recommender import recommend_movies

# Load data
movies, tags, scores = load_data()

# Step 1: Embeddings
print("\n✅ Computing SVD embeddings...")
tag_embeddings_df = compute_tag_embeddings_svd(scores, tags)
tag_embeddings_df.to_csv("data/tag_embeddings.csv")

print("\n✅ Clustering tags...")
tag_embeddings_df = cluster_tags(tag_embeddings_df, n_clusters=20)
tag_embeddings_df.to_csv("data/tag_clusters.csv")

# Optional visualization
plot_tsne(tag_embeddings_df)

# Semantic outlier detection
print("\n🔍 Detecting outliers...")
tag_list = tags['tag'].tolist()
semantic_df = compute_tag_embeddings_semantic(tag_list)
semantic_df['cluster'] = tag_embeddings_df.loc[semantic_df.index, 'cluster']
outliers = find_outliers(semantic_df, n_clusters=20)
for cluster_id, items in outliers.items():
    print(f"\nCluster {cluster_id} Outliers:")
    for tag, dist in items:
        print(f"  {tag}: {dist:.3f}")

# Recommend movies
print("\n🎯 Recommending movies based on user mood...")
user_input = input("Describe your mood: ")
top_movies = recommend_movies(user_input, tags, scores, movies)
print("\n🎬 Top Movie Recommendations:")
for i, row in top_movies.iterrows():
    print(f"{row['title']} (score: {row['relevance']:.3f})")