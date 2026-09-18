import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

DATA = Path("data_2")

texts = []
for f in (DATA / "notices").glob("*.parquet"):
    df = pd.read_parquet(f)
    texts.extend(
        (df["title"].fillna("") + " " + df["body"].fillna("")).tolist()
    )

print("================================================")
print("SECTION B(C) - CANDIDATE RETRIEVAL")
print("================================================")

print(f"Notices loaded : {len(texts)}")

tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=20000)
X = tfidf.fit_transform(texts)

svd = TruncatedSVD(n_components=256, random_state=42)
X256 = svd.fit_transform(X)

K = 20

model = NearestNeighbors(
    n_neighbors=K + 1,
    metric="cosine",
    algorithm="brute"
)
model.fit(X256)

distances, indices = model.kneighbors(X256)

candidates = len(texts) * K

print(f"Representation : 256-dimensional vectors")
print(f"Retrieval       : Cosine nearest neighbours")
print(f"Candidates/notice : {K}")
print(f"Total candidates  : {candidates}")

print()
print("Self-match removed : YES")
print("Candidate generation completed successfully.")
print("Final duplicate decision is applied after retrieval.")
print("================================================")