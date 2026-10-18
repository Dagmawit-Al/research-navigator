import json
import pickle
import networkx as nx
from collections import defaultdict
from itertools import combinations
from pathlib import Path

def build_citation_graph(papers):
    """
    Builds a directed citation graph.
    Node = paper
    Edge A -> B = paper A cites paper B
    """
    print("Building citation graph...")
    G = nx.DiGraph()

    paper_lookup = {}
    for p in papers:
        pid = p.get("paperId")
        if not pid:
            continue
        
        paper_lookup[pid] = p
        G.add_node(pid, 
            title=p.get("title", ""),
            year=p.get("year", 0),
            citationCount=p.get("citationCount", 0),
            fieldsOfStudy=p.get("fieldsOfStudy", "")
        )

    # add edges
    for p in papers:
        pid = p.get("paperId")
        if not pid:
            continue
        for ref in (p.get("references") or []):
            ref_id = ref.get("paperId") if isinstance(ref, dict) else ref
            if ref_id and ref_id in paper_lookup:
                G.add_edge(pid, ref_id)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, paper_lookup

def get_graph_stats(G):
    print(f"\n--- Citation Graph Stats ---")
    print(f"Nodes (papers): {G.number_of_nodes()}")
    print(f"Edges (citations): {G.number_of_edges()}")
    
    # most cited papers in our dataset
    in_degrees = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nMost cited papers in dataset:")
    for node, degree in in_degrees:
        print(f"  {degree} citations: {G.nodes[node].get('title', '')[:60]}")

def extract_topics(papers):
    """
    Fine-grained topic extraction using TF-IDF/KMeans clusters
    (see topics.py). Falls back to fieldsOfStudy if not available.
    """
    topic_papers = defaultdict(set)
    
    for p in papers:
        pid = p.get("paperId")
        
        # use fine-grained topic if available
        if "fine_topic_name" in p:
            topic_papers[p["fine_topic_name"]].add(pid)
        else:
            fields = p.get("fieldsOfStudy") or []
            if isinstance(fields, str):
                fields = [f.strip() for f in fields.split(",") if f.strip()]
            for field in fields:
                if field:
                    topic_papers[field].add(pid)
    
    return topic_papers

def detect_research_gaps(papers, G, min_topic_papers=30, top_n=10):
    """
    Core research gap detection logic.
    
    Idea: if two topics rarely appear together in the same papers,
    and papers combining them have low citation counts,
    that's a potential research gap.
    """
    print("\nDetecting research gaps...")
    topic_papers = extract_topics(papers)
    
    # filter topics with enough papers
    valid_topics = {
        t: pids for t, pids in topic_papers.items() 
        if len(pids) >= min_topic_papers
    }
    
    print(f"Valid topics found: {list(valid_topics.keys())}")
    
    gaps = []
    topic_list = list(valid_topics.keys())
    
    for t1, t2 in combinations(topic_list, 2):
        papers_t1 = valid_topics[t1]
        papers_t2 = valid_topics[t2]
        
        # papers that cover BOTH topics
        overlap = papers_t1 & papers_t2
        
        # overlap ratio relative to smaller topic
        smaller = min(len(papers_t1), len(papers_t2))
        overlap_ratio = len(overlap) / smaller if smaller > 0 else 0
        
        # avg citation count of overlap papers
        avg_citations = 0
        if overlap:
            citations = [
                G.nodes[pid].get("citationCount", 0) 
                for pid in overlap 
                if pid in G.nodes
            ]
            avg_citations = sum(citations) / len(citations) if citations else 0
        
        # gap score: low overlap + low citations = underexplored
        gap_score = (1 - overlap_ratio) * (1 / (avg_citations + 1))
        
        gaps.append({
            "topic1": t1,
            "topic2": t2,
            "overlap_count": len(overlap),
            "overlap_ratio": round(overlap_ratio, 3),
            "avg_citations": round(avg_citations, 1),
            "gap_score": round(gap_score, 4),
            "example_papers": list(overlap)[:3]
        })
    
    # sort by gap score descending
    gaps = sorted(gaps, key=lambda x: x["gap_score"], reverse=True)
    return gaps[:top_n]

def find_collaborators(author_name, papers, top_n=5):
    """
    Layer 3: find researchers with complementary expertise.
    Based on topic overlap and citation relationships.
    """
    # collect topics of target author
    author_topics = defaultdict(int)
    author_papers = []
    
    for p in papers:
        authors = p.get("authors") or []
        author_names = [a.get("name", "") if isinstance(a, dict) else a for a in authors]
        
        if any(author_name.lower() in name.lower() for name in author_names):
            author_papers.append(p.get("paperId"))
            fields = p.get("fieldsOfStudy") or []
            if isinstance(fields, str):
                fields = [f.strip() for f in fields.split(",") if f.strip()]
            for f in fields:
                author_topics[f] += 1
    
    if not author_papers:
        return [], {}
    
    # find other authors who work in complementary but different topics
    other_authors = defaultdict(lambda: {"topics": defaultdict(int), "papers": []})
    
    for p in papers:
        pid = p.get("paperId")
        authors = p.get("authors") or []
        fields = p.get("fieldsOfStudy") or []
        if isinstance(fields, str):
            fields = [f.strip() for f in fields.split(",") if f.strip()]
        
        for a in authors:
            name = a.get("name", "") if isinstance(a, dict) else a
            if author_name.lower() not in name.lower() and name:
                for f in fields:
                    other_authors[name]["topics"][f] += 1
                other_authors[name]["papers"].append(pid)
    
    # score by complementarity
    scores = []
    for name, data in other_authors.items():
        other_topics = set(data["topics"].keys())
        my_topics = set(author_topics.keys())
        
        shared = my_topics & other_topics
        unique_to_other = other_topics - my_topics
        
        # good collaborator: shares some topics but brings new ones
        score = len(shared) * 0.5 + len(unique_to_other) * 1.0
        
        if score > 0:
            scores.append({
                "name": name,
                "score": round(score, 2),
                "shared_topics": list(shared),
                "unique_topics": list(unique_to_other)[:5],
                "paper_count": len(data["papers"])
            })
    
    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return scores[:top_n], author_topics

def save_graph(G, path="models/citation_graph.pkl"):
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")

def load_graph(path="models/citation_graph.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    with open("data/raw_papers_with_topics.json") as f:
        papers = json.load(f)
    
    G, paper_lookup = build_citation_graph(papers)
    get_graph_stats(G)
    
    gaps = detect_research_gaps(papers, G)
    print("\n--- Top Research Gaps ---")
    for g in gaps[:5]:
        print(f"  {g['topic1']} x {g['topic2']}: {g['overlap_count']} papers overlap, gap score {g['gap_score']}")
    
    save_graph(G)
    
    # save paper lookup for other modules
    with open("models/paper_lookup.pkl", "wb") as f:
        pickle.dump(paper_lookup, f)
    
    print("\nGraph pipeline complete!")