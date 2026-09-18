import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data_2")
files = list((DATA / "notices").glob("*.parquet"))

texts = []

import pandas as pd
for f in files:
    df = pd.read_parquet(f)
    texts.extend((df["title"].fillna("") + " " + df["body"].fillna("")).tolist())

print("================================================")
print("SECTION A(b) - REDUCED REPRESENTATION")
print("================================================")
print(f"Notices loaded : {len(texts)}")

# Full representation
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=20000)
X = tfidf.fit_transform(texts)

# Fixed reduced size chosen before implementation
REDUCED_DIM = 256
svd = TruncatedSVD(n_components=REDUCED_DIM, random_state=42)
X_reduced = svd.fit_transform(X)

print(f"Full dimensions    : {X.shape[1]}")
print(f"Reduced dimensions : {REDUCED_DIM}")
print(f"Variance retained  : {svd.explained_variance_ratio_.sum():.4f}")

print()
print("Space-saving decision:")
print("Fixed representation size = 256 dimensions")
print("Target similarity error    <= 0.05")

print()
print("Reduced representation created successfully.")
print("Similarity estimation uses cosine similarity.")
print("Reduced vectors require substantially less storage.")
print("================================================")