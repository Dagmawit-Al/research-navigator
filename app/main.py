import sys
import json
import pickle
import numpy as np
import faiss
import streamlit as st
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append("src")

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Navigator",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* General Application Canvas Background */
[data-testid="stAppViewContainer"] { background-color: #0D1B2A; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Target both the outer structural sidebar and the inner user content area */
[data-testid="stSidebar"], 
[data-testid="stSidebarUserContent"] { 
    background-color: #0A1520 !important; 
    border-right: 1px solid #1E3A4A; 
}

/* Ensure all nested typography inherits the correct neon/slate text elements */
[data-testid="stSidebarUserContent"] * { 
    color: #CBD5E1 !important; 
}

/* Clear default layout footprints cleanly without dropping chevron buttons */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #132338;
    border: 1px solid #1E3A4A;
    border-radius: 12px;
    padding: 16px;
}

/* Paper cards */
.paper-card {
    background: #132338;
    border: 1px solid #1E3A4A;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.paper-card:hover { border-color: #00A896; }
.paper-title {
    font-size: 16px;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 8px;
    line-height: 1.4;
}
.paper-meta {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
}
.badge {
    background: #1E3A4A;
    color: #94A3B8;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}
.badge-teal {
    background: #003D38;
    color: #00A896;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.score-bar-bg {
    background: #1E3A4A;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin: 6px 0 10px 0;
}
.score-bar-fill {
    background: linear-gradient(90deg, #00A896, #02C39A);
    border-radius: 4px;
    height: 6px;
}
.abstract-text {
    color: #94A3B8;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 8px;
}
.paper-link {
    color: #00A896;
    font-size: 13px;
    text-decoration: none;
    font-weight: 500;
}

/* Gap cards */
.gap-card {
    background: #132338;
    border: 1px solid #1E3A4A;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.gap-title {
    font-size: 15px;
    font-weight: 700;
    color: #02C39A;
    margin-bottom: 6px;
}
.gap-insight {
    color: #94A3B8;
    font-size: 13px;
    margin-top: 8px;
    padding: 8px 12px;
    background: #0D2535;
    border-left: 3px solid #00A896;
    border-radius: 0 8px 8px 0;
}

/* Collab cards */
.collab-card {
    background: #132338;
    border: 1px solid #1E3A4A;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}
.collab-name {
    font-size: 15px;
    font-weight: 700;
    color: #F1F5F9;
}
.collab-score {
    font-size: 20px;
    font-weight: 800;
    color: #00A896;
}

/* Section headers */
.section-header {
    font-size: 26px;
    font-weight: 800;
    color: #F1F5F9;
    margin-bottom: 4px;
}
.section-sub {
    color: #64748B;
    font-size: 14px;
    margin-bottom: 24px;
}

/* Hero */
.hero-box {
    background: linear-gradient(135deg, #132338 0%, #0A1F35 100%);
    border: 1px solid #1E3A4A;
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
}
.hero-title {
    font-size: 42px;
    font-weight: 900;
    color: #F1F5F9;
    line-height: 1.15;
    margin-bottom: 12px;
}
.hero-sub {
    font-size: 17px;
    color: #94A3B8;
    line-height: 1.6;
    max-width: 600px;
    margin-bottom: 28px;
}
.stat-row {
    display: flex;
    gap: 32px;
    flex-wrap: wrap;
}
.stat-item { text-align: left; }
.stat-number { font-size: 28px; font-weight: 800; color: #00A896; }
.stat-label { font-size: 12px; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }

/* Input styling */
.stTextArea textarea, .stTextInput input {
    background: #132338 !important;
    border: 1px solid #1E3A4A !important;
    border-radius: 10px !important;
    color: #F1F5F9 !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #00A896 !important;
    box-shadow: 0 0 0 2px rgba(0,168,150,0.15) !important;
}

/* Button */
.stButton > button {
    background: #00A896 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    font-size: 15px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #02C39A !important; }
</style>
""", unsafe_allow_html=True)

# ── load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("allenai-specter")

@st.cache_resource
def load_index():
    index = faiss.read_index("models/papers.index")
    with open("models/paper_ids.pkl", "rb") as f:
        ids = pickle.load(f)
    return index, ids

@st.cache_resource
def load_papers():
    with open("data/raw_papers_with_topics.json") as f:
        papers = json.load(f)
    return {p["paperId"]: p for p in papers if p.get("paperId")}

@st.cache_resource
def load_graph_obj():
    with open("models/citation_graph.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_gaps():
    sys.path.append("src")
    from graph import detect_research_gaps, build_citation_graph
    with open("data/raw_papers_with_topics.json") as f:
        papers = json.load(f)
    G, _ = build_citation_graph(papers)
    return detect_research_gaps(papers, G)

try:
    model       = load_model()
    index, ids  = load_index()
    papers_lookup = load_papers()
    artifacts_ready = True
except Exception as e:
    artifacts_ready = False

# ── helpers ────────────────────────────────────────────────────────────────────
def recommend_papers(query, profile, top_k=8):
    emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    
    # Fetch an expanded candidate pool to give re-ranking room to breathe
    scores, indices = index.search(emb, max(top_k * 6, 50))
    results = []
    
    for score, idx in zip(scores[0], indices[0]):
        pid = ids[idx]
        paper = papers_lookup.get(pid)
        if paper and paper.get("title"):
            base_score = float(score)
            boost = 0.0
            
            # Extract high-level metadata strings
            fields = paper.get("fieldsOfStudy") or []
            fields_str = " ".join(fields).lower() if isinstance(fields, list) else str(fields).lower()
            topic_str = str(paper.get("fine_topic_name", "")).lower()
            title_str = str(paper.get("title", "")).lower()
            
            # STRICT MATCHING: Abstract text is completely ignored here
            match = any(
                interest.lower() in title_str or 
                interest.lower() in topic_str or
                interest.lower() in fields_str
                for interest in profile["interests"]
            )
            
            # Apply the re-ranking premium if it passes the strict gate
            if match:
                boost += 0.15
                
            # Bias Style Adjustments
            year = paper.get("year") or 2000
            cites = paper.get("citationCount") or 0
            
            if profile["preference"] == "Recent Innovations":
                boost += min(1.0, max(0.0, (year - 2000) / 26.0)) * 0.05
            elif profile["preference"] == "Highly Cited":
                boost += min(1.0, np.log10(cites + 1) / 4.0) * 0.05
                
            results.append({
                **paper, 
                "score": base_score + boost,
                "interest_match": match
            })
            
    # Hard Metric Filter
    if profile.get("min_citations", 0) > 0:
        results = [r for r in results if (r.get("citationCount") or 0) >= profile["min_citations"]]
        
    # Re-sort using updated contextual score weights
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def render_paper_card(r, rank):
    score = r.get("score", 0)
    title = r.get("title", "Untitled")
    year  = r.get("year", "N/A")
    cites = r.get("citationCount", 0) or 0
    field = r.get("fine_topic_name") or ""
    if isinstance(field, list): field = field[0] if field else ""
    abstract = (r.get("abstract") or "")[:280] + "..."
    pid   = r.get("paperId", "")
    link  = f"https://www.semanticscholar.org/paper/{pid}" if pid else "#"
    bar_w = max(0, min(100, int((score - 0.5) / 0.5 * 100)))
    
    # Generate badges safely as complete strings or empty targets
    match_badge = "<span class='badge-teal'>★ Profile Match</span>" if r.get("interest_match") else ""
    field_badge = f"<span class='badge'>{field}</span>" if field else ""

    # CRITICAL: Keep HTML blocks flushed left to prevent Markdown indentation bugs
    html_content = f"""
<div class="paper-card">
<div class="paper-title">{rank}. {title}</div>
<div class="paper-meta">
<span class="badge-teal" style="background:#00A896; color:white;">Match Score: {score:.3f}</span>
{match_badge}
<span class="badge">📅 {year}</span>
<span class="badge">📖 {cites:,} citations</span>
{field_badge}
</div>
<div class="score-bar-bg"><div class="score-bar-fill" style="width:{bar_w}%"></div></div>
<div class="abstract-text">{abstract}</div>
<a href="{link}" target="_blank" class="paper-link">View on Semantic Scholar →</a>
</div>
"""
    st.markdown(html_content, unsafe_allow_html=True)

def render_gap_card(g, rank):
    t1 = g["topic1"].split(" / ")[0].title()
    t2 = g["topic2"].split(" / ")[0].title()
    st.markdown(f"""
    <div class="gap-card">
        <div class="gap-title">💡 {t1} × {t2}</div>
        <div class="paper-meta">
            <span class="badge">Gap Score: {g['gap_score']}</span>
            <span class="badge">{g['overlap_count']} papers overlap</span>
            <span class="badge">Avg citations: {g['avg_citations']}</span>
        </div>
        <div class="gap-insight">
            Only <strong>{g['overlap_count']}</strong> papers combine these two areas — 
            this intersection may represent an underexplored research opportunity.
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_collab_card(c, rank):
    shared = ", ".join(c["shared_topics"][:3]) or "—"
    unique = ", ".join(c["unique_topics"][:3]) or "—"
    st.markdown(f"""
    <div class="collab-card">
        <div>
            <div class="collab-name">{rank}. {c['name']}</div>
            <div style="color:#64748B;font-size:13px;margin-top:4px">
                <span style="color:#94A3B8">Shared:</span> {shared}<br>
                <span style="color:#94A3B8">Brings:</span> {unique}
            </div>
            <div style="margin-top:8px"><span class="badge">{c['paper_count']} papers in dataset</span></div>
        </div>
        <div class="collab-score">{c['score']}</div>
    </div>
    """, unsafe_allow_html=True)

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-size:32px; margin-bottom:8px;">🧭</div>
        <div style="font-size:20px; font-weight:800; color:#F1F5F9;">Research Navigator</div>
        <div style="font-size:12px; color:#64748B; margin-top:4px;">Academic Discovery System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    mode = st.radio(
        "Navigation",
        ["🏠  Home", "📄  Paper Recommender"],
        label_visibility="collapsed"
        )
    st.markdown("---")

#     mode = st.radio(
#     "Navigation",
#     ["🏠  Home", "📄  Paper Recommender", "🔍  Research Gaps", "🤝  Collaborator Finder"],
#     label_visibility="collapsed"
# )
    # Interactive User Profile Management Layer
    st.markdown("<div style='font-size:14px; font-weight:700; color:#F1F5F9; margin-bottom:12px;'>👤 Research Profile Context</div>", unsafe_allow_html=True)
    
    user_profile = {
        "interests": st.multiselect(
            "Target Subfields", 
            ["Recommender Systems", "Graph Neural Networks", "Natural Language Processing", "Collaborative Filtering", "Computer Vision", "Federated Learning", "Explainable AI"], 
            default=["Recommender Systems"]
        ),
        "preference": st.radio(
            "Ranking Optimization style", 
            ["Balanced Strategy", "Recent Innovations", "Highly Cited"]
        ),
        "min_citations": st.number_input(
            "Minimum Citation Filter", 
            min_value=0, 
            value=0, 
            step=10
        )
    }
    
    st.markdown("---")

    if artifacts_ready:
        n_papers = len(papers_lookup)
        st.markdown(f"""
        <div style="font-size:12px; color:#64748B; line-height:2;">
            📦 <b style="color:#94A3B8">{n_papers:,}</b> papers indexed<br>
            🧠 SPECTER embeddings<br>
            ⚡ FAISS similarity search<br>
            🕸️ NetworkX citation graph
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Models not ready. Run the pipeline first.")

# ── home ───────────────────────────────────────────────────────────────────────
if mode == "🏠  Home":
    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">🧭 Research Navigator</div>
        <div class="hero-sub">
            A graph-based academic discovery system that helps researchers find relevant papers, 
            uncover research gaps, and discover potential collaborators — all in one place.
        </div>
        <div class="stat-row">
            <div class="stat-item">
                <div class="stat-number">2,975</div>
                <div class="stat-label">Papers indexed</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">3</div>
                <div class="stat-label">Discovery features</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">15</div>
                <div class="stat-label">Topic clusters</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">768</div>
                <div class="stat-label">Embedding dimensions</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # feature cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="gap-card">
            <div style="font-size:28px;margin-bottom:10px;">📄</div>
            <div class="gap-title">Paper Recommender</div>
            <div class="abstract-text">Enter a topic, title, or abstract. Get semantically similar papers ranked by SPECTER embedding similarity.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="gap-card">
            <div style="font-size:28px;margin-bottom:10px;">🔍</div>
            <div class="gap-title">Research Gap Detector</div>
            <div class="abstract-text">Discover underexplored intersections between research topics using citation graph analysis and topic clustering.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="gap-card">
            <div style="font-size:28px;margin-bottom:10px;">🤝</div>
            <div class="gap-title">Collaborator Finder</div>
            <div class="abstract-text">Find researchers with complementary expertise. Scored by unique topics they bring to the table.</div>
        </div>
        """, unsafe_allow_html=True)

    # citation distribution chart
    if artifacts_ready:
        st.markdown("<div style='margin-top:32px;'><div class='section-header'>📊 Dataset Overview</div><div class='section-sub'>Citation distribution across indexed papers</div></div>", unsafe_allow_html=True)

        papers_list = list(papers_lookup.values())
        years = [p.get("year") for p in papers_list if p.get("year") and p.get("year") >= 2000]
        year_counts = pd.Series(years).value_counts().sort_index()

        st.bar_chart(
            year_counts.rename("Papers"),
            color="#00A896",
            use_container_width=True,
            height=220
        )

# ── paper recommender ──────────────────────────────────────────────────────────
elif mode == "📄  Paper Recommender":
    st.markdown("<div class='section-header'>📄 Paper Recommender</div><div class='section-sub'>Enter a research topic, paper title, or abstract to find semantically similar papers.</div>", unsafe_allow_html=True)

    query = st.text_area("Your query", placeholder="e.g. 'graph neural networks for collaborative filtering in recommender systems'", height=100, label_visibility="collapsed")
    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.slider("Results", 3, 15, 8, label_visibility="collapsed")
    with col2:
        st.markdown(f"<div style='padding-top:8px;color:#64748B;font-size:13px;'>Showing top {top_k} results tailored to your profile matrix</div>", unsafe_allow_html=True)

    if st.button("🔍 Find Papers", disabled=not artifacts_ready):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching through embeddings and adjusting weights..."):
                results = recommend_papers(query, user_profile, top_k)
            st.markdown(f"<div style='color:#00A896;font-weight:600;margin-bottom:16px;'>Found {len(results)} matches meeting criteria</div>", unsafe_allow_html=True)
            for i, r in enumerate(results):
                render_paper_card(r, i + 1)

# ── research gaps ──────────────────────────────────────────────────────────────
elif mode == "🔍  Research Gaps":
    st.markdown("<div class='section-header'>🔍 Research Gap Detector</div><div class='section-sub'>Discover underexplored intersections between research topics using citation graph analysis.</div>", unsafe_allow_html=True)

    if st.button("⚡ Detect Gaps", disabled=not artifacts_ready):
        with st.spinner("Analysing citation graph and topic clusters..."):
            gaps = load_gaps()

        if not gaps:
            st.info("Not enough topic overlap data.")
        else:
            st.markdown(f"<div style='color:#00A896;font-weight:600;margin-bottom:16px;'>Found {len(gaps)} potential research gaps</div>", unsafe_allow_html=True)

            # mini chart: gap scores
            gap_df = pd.DataFrame([{
                "Topics": f"{g['topic1'].split('/')[0].strip()[:15]} × {g['topic2'].split('/')[0].strip()[:15]}",
                "Gap Score": g["gap_score"]
            } for g in gaps[:8]])

            st.bar_chart(gap_df.set_index("Topics")["Gap Score"], color="#00A896", height=200, use_container_width=True)

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            for i, g in enumerate(gaps):
                render_gap_card(g, i + 1)

# ── collaborator finder ────────────────────────────────────────────────────────
elif mode == "🤝  Collaborator Finder":
    st.markdown("<div class='section-header'>🤝 Collaborator Finder</div><div class='section-sub'>Find researchers with complementary expertise. Unique topics score higher than shared ones.</div>", unsafe_allow_html=True)

    author_name = st.text_input("Author name", placeholder="e.g. Xiangnan He", label_visibility="collapsed")

    if st.button("🔎 Find Collaborators", disabled=not artifacts_ready):
        if not author_name.strip():
            st.warning("Please enter an author name.")
        else:
            with st.spinner("Analysing author network..."):
                sys.path.append("src")
                from graph import find_collaborators
                papers_list = list(papers_lookup.values())
                collabs, my_topics = find_collaborators(author_name.strip(), papers_list)

            if not collabs:
                st.warning(f"Author '{author_name}' not found. Try: D. Jannach, Xiangnan He, F. Ricci")
            else:
                topics_str = ", ".join(list(my_topics.keys())[:5])
                st.markdown(f"""
                <div style="background:#132338;border:1px solid #1E3A4A;border-radius:12px;padding:16px 20px;margin-bottom:20px;">
                    <div style="color:#64748B;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Research profile</div>
                    <div style="color:#F1F5F9;font-size:15px;font-weight:700;">{author_name}</div>
                    <div style="color:#94A3B8;font-size:13px;margin-top:4px;">Topics: {topics_str}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"<div style='color:#00A896;font-weight:600;margin-bottom:16px;'>Found {len(collabs)} potential collaborators</div>", unsafe_allow_html=True)

                for i, c in enumerate(collabs):
                    render_collab_card(c, i + 1)