import json
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "allenai-specter"  
BATCH_SIZE = 32

def load_texts(path="data/raw_papers.json"):
    with open(path) as f:
        papers = json.load(f)
    
    valid = []
    for p in papers:
        if p.get("title") and p.get("abstract"):
            valid.append({
                "paperId": p["paperId"],
                "title": p["title"],
                "abstract": p["abstract"],
                "text": f"{p['title']}. {p['abstract']}"
            })
    
    print(f"Valid papers for embedding: {len(valid)}")
    return valid

def generate_embeddings(papers):
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    texts = [p["text"] for p in papers]
    ids = [p["paperId"] for p in papers]
    
    print(f"Generating embeddings for {len(texts)} papers...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True 
    )
    
    return ids, embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    print(f"Building FAISS index, dimension: {dim}")
    
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    print(f"Index built with {index.ntotal} vectors")
    return index

def save_artifacts(ids, embeddings, index):
    Path("models").mkdir(exist_ok=True)
    
    faiss.write_index(index, "models/papers.index")
    np.save("models/embeddings.npy", embeddings)
    
    with open("models/paper_ids.pkl", "wb") as f:
        pickle.dump(ids, f)
    
    print("Saved: models/papers.index, models/embeddings.npy, models/paper_ids.pkl")

def load_artifacts():
    index = faiss.read_index("models/papers.index")
    embeddings = faiss.read_index("models/papers.index")
    embeddings = np.load("models/embeddings.npy")
    
    with open("models/paper_ids.pkl", "rb") as f:
        ids = pickle.load(f)
    
    return ids, embeddings, index

def recommend_similar(query_text, index, ids, papers_lookup, top_k=10):
    model = SentenceTransformer(MODEL_NAME)
    
    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    scores, indices = index.search(query_embedding, top_k + 1)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        paper_id = ids[idx]
        paper = papers_lookup.get(paper_id, {})
        if paper:
            results.append({
                "paperId": paper_id,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", "")[:200] + "...",
                "year": paper.get("year", ""),
                "score": float(score)
            })
    
    return results[1:]  

if __name__ == "__main__":
    papers = load_texts()
    ids, embeddings = generate_embeddings(papers)
    index = build_faiss_index(embeddings)
    save_artifacts(ids, embeddings, index)
    print("\nEmbedding pipeline complete!")