import sqlite3
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to python path to load src
sys.path.append("/Users/vladimirkasterin/python/graph/back")
from src.config import config
from src.vector_search import VectorSearch

query = "Каким образом ограничения в оценке качества долгосрочной памяти (Challenge IV) соотносятся с наблюдаемой немонотонной эффективностью метода Self-Reflection при увеличении фактора ветвления (Branching Factor K)?"

print("Configured Embedding Model:", config.data["embedding"]["model_name"])

# Initialize VectorSearch
vs = VectorSearch()
# Get embedding for query
emb = vs.get_embedding(query, is_query=True)
emb_arr = np.array(emb, dtype=np.float32)

db_path = "/Users/vladimirkasterin/.local/share/pdf-graph-analyzer/graph.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# Let's perform the vector search manually using cosine similarity
# Cosine similarity = dot(A, B) / (norm(A) * norm(B))
# Assuming embeddings are normalized, it is just dot(A, B)

print("\n--- Running direct vector search manually ---")
cursor = conn.execute("SELECT id, paper_id, text_content, embedding FROM chunks")
chunks = cursor.fetchall()

scored = []
for c in chunks:
    c_emb = np.frombuffer(c['embedding'], dtype=np.float32)
    # Cosine similarity
    similarity = np.dot(emb_arr, c_emb) / (np.linalg.norm(emb_arr) * np.linalg.norm(c_emb))
    scored.append((c['id'], c['paper_id'], c['text_content'], float(similarity)))

scored.sort(key=lambda x: x[3], reverse=True)

for i, (cid, pid, text, sim) in enumerate(scored[:20], 1):
    print(f"{i}. Similarity: {sim:.4f} | Paper: {pid} | Chunk: {cid} | Snippet: {text[:150]}...")

print("\n--- Top chunks for context_engineering ---")
ce_scored = [x for x in scored if x[1] == 'context_engineering_20_the_context_of_context_engineering']
for i, (cid, pid, text, sim) in enumerate(ce_scored[:5], 1):
    print(f"CE {i}. Similarity: {sim:.4f} | Chunk: {cid} | Snippet: {text[:150]}...")
