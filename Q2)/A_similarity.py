import re
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load notices
files = glob.glob("data/notices/*.parquet")
notices = pd.concat(
    [pd.read_parquet(f) for f in files],
    ignore_index=True
)

# Normalize notice text
def clean_text(text):
    text = str(text).lower()

    # Remove dates
    text = re.sub(r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b', ' ', text)

    # Remove monetary amounts
    text = re.sub(r'(₹|rs\.?|inr)\s*[\d,]+(?:\.\d+)?', ' ', text)

    # Remove tender/reference numbers
    text = re.sub(
        r'\b(?:tender|reference|ref|bid|notice)[-_/: ]?[a-z0-9/-]*\d[a-z0-9/-]*\b',
        ' ',
        text
    )

    # Remove numbers
    text = re.sub(r'\b\d+\b', ' ', text)

    # Keep alphabetic text
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


notices["clean_text"] = (
    notices["title"].fillna("") + " " +
    notices["body"].fillna("")
).apply(clean_text)


# Two competing choices
word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(3, 3)
)

character = TfidfVectorizer(
    analyzer="char",
    ngram_range=(5, 5)
)

word_matrix = word.fit_transform(notices["clean_text"])
char_matrix = character.fit_transform(notices["clean_text"])


# Take one same and one different pair from labelled data
pairs = pd.read_csv("data/labeled_pairs.csv")

print("=" * 65)
print("SECTION A(a) - SIMILARITY DEFINITION")
print("=" * 65)

print("\nCorpus:")
print(f"Notices loaded : {len(notices):,}")
print(f"Labelled pairs : {len(pairs):,}")

print("\nSimilarity score:")
print("Cosine similarity between TF-IDF vectors")

print("\nText decomposition:")
print("Choice 1 : Word 3-grams")
print("Choice 2 : Character 5-grams")

print("\nNoise removed:")
print("Dates, monetary amounts, reference numbers, standalone numbers")

print("\nSignal retained:")
print("Tender title and body wording")

print("\nVector dimensions:")
print(f"Word 3-grams      : {word_matrix.shape[1]:,}")
print(f"Character 5-grams : {char_matrix.shape[1]:,}")

print("\nChosen design:")
print("Word 3-gram TF-IDF + cosine similarity")

print("\nReason:")
print("Captures meaningful multi-word tender phrases while")
print("reducing unstable dates, amounts and portal references.")

print("=" * 65)