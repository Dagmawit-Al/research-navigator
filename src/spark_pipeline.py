import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from spark_config import get_spark_session
from pyspark.sql import functions as F

def load_papers(spark, path=None):
    if path is None:
        path = ROOT / "data" / "raw_papers.json"
    
    with open(path) as f:
        papers = json.load(f)

    flat = []
    for p in papers:
        flat.append({
            "paperId": p.get("paperId", ""),
            "title": p.get("title", "") or "",
            "abstract": p.get("abstract", "") or "",
            "year": p.get("year", 0) or 0,
            "citationCount": p.get("citationCount", 0) or 0,
            "fieldsOfStudy": ", ".join(p.get("fieldsOfStudy") or []),
            "authors": ", ".join([a.get("name", "") for a in (p.get("authors") or [])]),
            "references": [r.get("paperId") for r in (p.get("references") or []) if r.get("paperId")],
            "citations": [c.get("paperId") for c in (p.get("citations") or []) if c.get("paperId")],
        })

    return spark.createDataFrame(flat)

def clean_text(df):
    df = df.filter(F.col("title") != "")
    df = df.filter(F.col("abstract") != "")
    df = df.filter(F.col("year") >= 2000)
    df = df.withColumn(
        "text",
        F.concat(F.col("title"), F.lit(". "), F.col("abstract"))
    )
    return df

def basic_stats(df):
    print(f"Total papers: {df.count()}")
    print(f"Year range: {df.agg(F.min('year'), F.max('year')).collect()}")
    print(f"Avg citations: {df.agg(F.avg('citationCount')).collect()}")
    df.groupBy("fieldsOfStudy").count().orderBy(F.desc("count")).show(10)

if __name__ == "__main__":
    spark = get_spark_session()

    print("Loading papers...")
    df = load_papers(spark)

    print("Cleaning...")
    df = clean_text(df)

    print("\n--- Dataset Stats ---")
    basic_stats(df)

    output_path = str(ROOT / "data" / "papers.parquet")
    df.drop("references", "citations").write.mode("overwrite").parquet(output_path)
    print(f"\nSaved to {output_path}")

    spark.stop()