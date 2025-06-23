import pandas as pd
import os

def load_data():
    data_dir = "data"
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    tags = pd.read_csv(os.path.join(data_dir, "genome-tags.csv"))
    scores = pd.read_csv(os.path.join(data_dir, "genome-scores.csv"))
    scores = scores.merge(tags, on="tagId")
    return movies, tags, scores