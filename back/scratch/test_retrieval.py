import sys
from pathlib import Path

# Add project root to python path to load src
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.container import container
from src.config import config

def test():
    # Configure B6 settings
    components = {k: True for k in config.rag_components.keys()}
    if "rag_components" not in config.data:
        config.data["rag_components"] = {}
    for k, v in components.items():
        config.data["rag_components"][k] = v
    config.data["llm"]["hyde_enabled"] = True

    rag = container.get_rag_service(use_cloud=False)
    
    query = "Каковы точные параметры архитектур моделей BERT_BASE и BERT_LARGE (количество слоев, размер скрытого состояния, число голов внимания, общее число параметров), и по какой причине размер BERT_BASE был сделан именно таким?"
    
    print("Running retrieve_relevant_chunks...")
    
    # Let's inspect retrieve_relevant_chunks steps
    # 1. Intent classifier
    clean_q, filters = rag._classify_intent_and_extract_filters(query)
    print(f"Clean Q: {clean_q}, Filters: {filters}")
    
    # 2. Query expansion
    expanded_queries = rag._expand_query(clean_q)
    print(f"Expanded Queries: {expanded_queries}")
    
    # 3. Dense search
    dense_limit = 5 * 2 if len(expanded_queries) == 1 else 5
    all_dense_results = {}
    for variant in expanded_queries:
        variant_emb = rag.emb_engine.get_embedding(variant)
        dense_res = rag.vector_repo.search_similar_chunks(variant_emb, limit=dense_limit)
        print(f"Variant '{variant}' dense results: {[(c.id, s) for c, s in dense_res[:3]]}")
        for chunk, score in dense_res:
            if chunk.id not in all_dense_results:
                all_dense_results[chunk.id] = (chunk, score)
            else:
                existing_chunk, existing_score = all_dense_results[chunk.id]
                if score > existing_score:
                    all_dense_results[chunk.id] = (chunk, score)
                    
    # 4. HyDE
    print("HyDE generation:")
    try:
        hypothetical = rag.llm_engine.generate_response(
            prompt=rag.llm_engine.extract_json(rag.llm_engine.generate_response(f"Выступи как HyDE генератор. Вопрос: {query}")) if hasattr(rag.llm_engine, "extract_json") else f"Hypothetical answer to: {query}",
            max_tokens=300
        )
        print(f"HyDE answer: {hypothetical[:100]}...")
    except Exception as e:
        print(f"HyDE error: {e}")
        
    dense_results = list(all_dense_results.values())
    dense_results.sort(key=lambda x: x[1], reverse=True)
    print(f"Total merged dense results: {len(dense_results)}")
    
    # 5. FTS5 search
    fts5_results = rag.vector_repo.search_text_fts5(clean_q, limit=10)
    print(f"FTS5 results count: {len(fts5_results)}")
    print(f"FTS5 top 3: {[(c.id, s) for c, s in fts5_results[:3]]}")
    
    # RRF & blending
    id_to_chunk = {}
    for chunk, _ in dense_results:
        id_to_chunk[chunk.id] = chunk
    for chunk, _ in fts5_results:
        id_to_chunk[chunk.id] = chunk
        
    fts_weight = 1.0
    if fts5_results:
        max_bm25 = max(score for _, score in fts5_results)
        if max_bm25 < 1.0:
            fts_weight = 0.2
        elif max_bm25 < 3.0:
            fts_weight = 0.5
            
    rrf_scores = {}
    for rank, (chunk, _) in enumerate(dense_results, start=1):
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 * (1.0 / (60.0 + rank))
    for rank, (chunk, _) in enumerate(fts5_results, start=1):
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + fts_weight * (1.0 / (60.0 + rank))
        
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    candidates = [id_to_chunk[cid] for cid in sorted_ids[:10] if cid in id_to_chunk]
    
    print("\nCandidates before reranking:")
    for c in candidates:
        print(f"  ID: {c.id}, RRF score: {rrf_scores[c.id]:.4f}, Title: {c.paper_id}")
        
    # Reranking
    reranker = rag._get_reranker()
    pairs = [(clean_q, c.text_content) for c in candidates]
    scores = reranker.predict(pairs)
    
    min_r = min(scores)
    max_r = max(scores)
    range_r = max_r - min_r if max_r > min_r else 1.0
    norm_r = [(s - min_r) / range_r for s in scores]
    
    rrf_vals = [rrf_scores[c.id] for c in candidates]
    min_rrf = min(rrf_vals)
    max_rrf = max(rrf_vals)
    range_rrf = max_rrf - min_rrf if max_rrf > min_rrf else 1.0
    norm_rrf = [(rrf_scores[c.id] - min_rrf) / range_rrf for c in candidates]
    
    scored_candidates = []
    for idx, c in enumerate(candidates):
        blended_score = 0.7 * norm_r[idx] + 0.3 * norm_rrf[idx]
        scored_candidates.append((c, blended_score, float(scores[idx]), norm_r[idx], norm_rrf[idx]))
        
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    print("\nReranked candidates:")
    for c, blended, raw, nr, nrrf in scored_candidates:
        print(f"  ID: {c.id}, Blended: {blended:.4f}, Raw rerank: {raw:.4f}, Norm rerank: {nr:.4f}, Norm RRF: {nrrf:.4f}, Title: {c.paper_id}")

if __name__ == "__main__":
    test()
