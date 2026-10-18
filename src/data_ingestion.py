import requests
import json
import time
import os
from pathlib import Path

BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,authors,citations,references,fieldsOfStudy,citationCount"
API_KEY = os.getenv("S2_API_KEY", "")

def get_headers():
    if API_KEY:
        return {"x-api-key": API_KEY}
    return {}

def fetch_papers(query, max_papers=1500):
    papers = []
    offset = 0
    limit = 100
    cache_file = Path(f"data/cache_{query.replace(' ', '_')}.json")

    if cache_file.exists():
        print(f"Cache hit for '{query}', loading from disk...")
        with open(cache_file) as f:
            return json.load(f)

    while len(papers) < max_papers:
        url = f"{BASE_URL}/paper/search"
        params = {
            "query": query,
            "fields": FIELDS,
            "limit": min(limit, max_papers - len(papers)),
            "offset": offset
        }

        try:
            response = requests.get(url, params=params, headers=get_headers())

            if response.status_code == 429: #rate limit
                wait = 60 if not API_KEY else 10
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code == 504: #server timeout
                print(f"504 timeout for '{query}', retrying in 15s...")
                time.sleep(15)
                continue

            if response.status_code != 200:
                print(f"Error {response.status_code} for query '{query}', stopping this query")
                break

            data = response.json()
            batch = data.get("data", [])

            if not batch:
                break

            papers.extend(batch)
            offset += len(batch)
            print(f"  [{query}] {len(papers)} papers fetched...")

            with open(cache_file, "w") as f:
                json.dump(papers, f)

            sleep_time = 1 if API_KEY else 3
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Exception: {e}, retrying in 10s...")
            time.sleep(10)

    # save whatever we got even if loop ended early
    if papers:
        with open(cache_file, "w") as f:
            json.dump(papers, f)

    return papers

def load_all_papers():
    queries = [
        "recommender systems",
        "graph neural networks",
        "natural language processing",
        "federated learning",
        "explainable artificial intelligence",
        "knowledge graph embedding",
        "collaborative filtering"
    ]

    all_papers = []
    for q in queries:
        print(f"\nFetching: {q}")
        papers = fetch_papers(q, max_papers=1500)
        all_papers.extend(papers)

    seen = set()
    unique = []
    for p in all_papers:
        pid = p.get("paperId")
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(p)

    print(f"\nTotal unique papers: {len(unique)}")

    with open("data/raw_papers.json", "w") as f:
        json.dump(unique, f)
    print("Saved to data/raw_papers.json")

    return unique

if __name__ == "__main__":
    papers = load_all_papers()