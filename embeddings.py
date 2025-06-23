import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

from utils import load_data

def compute_tag_embeddings_svd(scores, tags, n_components=50):
    tag_movie_matrix = scores.pivot(index='tagId', columns='movieId', values='relevance').fillna(0)
    normalized = normalize(tag_movie_matrix, axis=1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tag_embeddings = svd.fit_transform(normalized)

    tag_id_to_name = dict(zip(tags['tagId'], tags['tag']))
    tag_names = [tag_id_to_name[tag_id] for tag_id in tag_movie_matrix.index]
    df = pd.DataFrame(tag_embeddings, index=tag_names)
    df.columns = [f"dim_{i+1}" for i in range(n_components)]
    return df

def compute_tag_embeddings_semantic(tag_list):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(tag_list, show_progress_bar=True)
    return pd.DataFrame(embeddings, index=tag_list)