import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

N_CLUSTERS = 15

def load_papers(path="data/raw_papers.json"):
    with open(path) as f:
        papers = json.load(f)
    valid = [p for p in papers if p.get("title") and p.get("abstract")]
    return valid

def extract_topics(papers, n_clusters=N_CLUSTERS):
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]

    print(f"Vectorizing {len(texts)} papers with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.7
    )
    X = vectorizer.fit_transform(texts)

    print(f"Clustering into {n_clusters} topics with KMeans...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    feature_names = vectorizer.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    cluster_names = {}
    for i in range(n_clusters):
        top_terms = [feature_names[ind] for ind in order_centroids[i, :4]]
        cluster_names[i] = " / ".join(top_terms)
        print(f"  Cluster {i}: {cluster_names[i]}  ({(labels == i).sum()} papers)")

    for idx, p in enumerate(papers):
        p["fine_topic_id"] = int(labels[idx])
        p["fine_topic_name"] = cluster_names[labels[idx]]

    return papers, cluster_names

def save_results(papers, cluster_names):
    Path("models").mkdir(exist_ok=True)

    with open("data/raw_papers_with_topics.json", "w") as f:
        json.dump(papers, f)

    with open("models/cluster_names.pkl", "wb") as f:
        pickle.dump(cluster_names, f)

    print(f"\nSaved: data/raw_papers_with_topics.json, models/cluster_names.pkl")

if __name__ == "__main__":
    papers = load_papers()
    papers, cluster_names = extract_topics(papers)
    save_results(papers, cluster_names)
    print("\nTopic extraction complete!")