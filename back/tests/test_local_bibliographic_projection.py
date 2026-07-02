import sys
from unittest.mock import MagicMock


import os
import math
import sqlite3
import socket
import pytest
from src.models import Paper
from src.repository.sqlite_impl import SQLiteGraphRepository
from src.services.bibliographic import (
    canonicalize_reference,
    BibliographicProjectionService
)
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
from scripts.migrate_local_bibliographic_projection import run_migration

# 13.1. Unit tests: canonicalization
def test_canonicalize_reference_doi():
    # DOI prefix and casing normalization
    ref1 = canonicalize_reference("Check out this paper https://doi.org/10.1234/ABC")
    assert ref1.doi == "10.1234/abc"
    assert ref1.work_id == "work:doi:10.1234/abc"
    assert ref1.canonicalization_method == "doi"

    ref2 = canonicalize_reference("doi:10.1234/ABC")
    assert ref2.doi == "10.1234/abc"
    assert ref2.work_id == "work:doi:10.1234/abc"

def test_canonicalize_reference_arxiv():
    # arXiv ID casing and prefix normalization
    ref = canonicalize_reference("Refer to arXiv:1706.03762v1")
    assert ref.arxiv_id == "1706.03762"
    assert ref.work_id == "work:arxiv:1706.03762"
    assert ref.canonicalization_method == "arxiv"

def test_canonicalize_reference_url():
    # URL normalization
    ref = canonicalize_reference("Available at https://arxiv.org/abs/1706.03762")
    # Wait, https://arxiv.org/abs/1706.03762 is matched by arXiv regex!
    assert ref.arxiv_id == "1706.03762"
    
    # Let's test a plain website URL
    ref_url = canonicalize_reference("Check https://google.com/path/to/resource/")
    assert ref_url.url == "https://google.com/path/to/resource"
    assert ref_url.work_id.startswith("work:url:")
    assert ref_url.canonicalization_method == "url"

def test_canonicalize_reference_title():
    # Title fallback normalization
    ref = canonicalize_reference("", {"title": "Attention Is All You Need", "year": 2017})
    assert ref.normalized_title == "attention is all you need"
    assert ref.work_id.startswith("work:title:")
    assert ref.canonicalization_method == "normalized_title_hash"

def test_canonicalize_reference_raw_fallback():
    # Raw fallback normalization
    ref = canonicalize_reference("A very random citation string with no identifiers")
    assert ref.work_id.startswith("work:raw:")
    assert ref.canonicalization_method == "raw_hash"

def test_canonicalize_reference_determinism():
    # Determinism check
    ref1 = canonicalize_reference("https://doi.org/10.1234/ABC")
    ref2 = canonicalize_reference("https://doi.org/10.1234/ABC")
    assert ref1.work_id == ref2.work_id

# 13.2 & 13.3. Unit tests: reference vector math and zero norm cases
def test_reference_vector_math_synthetic_corpus():
    # Synthetic corpus math check:
    # Paper A cites R1, R2
    # Paper B cites R1, R3
    # Paper C cites R4
    # N = 3
    # df(R1) = 2, df(R2) = 1, df(R3) = 1, df(R4) = 1
    # idf(R1) = log(3/2) = 0.405465108
    # idf(R2) = log(3/1) = 1.098612289
    # idf(R3) = log(3/1) = 1.098612289
    # idf(R4) = log(3/1) = 1.098612289
    
    # Vector norms:
    # v_A = [idf(R1), idf(R2)] => norm_A = sqrt(0.405465**2 + 1.098612**2) = 1.1710188
    # v_B = [idf(R1), idf(R3)] => norm_B = sqrt(0.405465**2 + 1.098612**2) = 1.1710188
    # dot_product = idf(R1)**2 = 0.164402
    # cosine_similarity = dot_product / (norm_A * norm_B) = 0.164402 / 1.371285 = 0.119889
    
    repo = MagicMock()
    
    paper_A = Paper(id="A", title="Paper A")
    paper_B = Paper(id="B", title="Paper B")
    paper_C = Paper(id="C", title="Paper C")
    
    repo.get_non_placeholder_papers.return_value = [paper_A, paper_B, paper_C]
    
    # Mock edges list
    edges = [
        ("A", "R1", "CITES", {}),
        ("A", "R2", "CITES", {}),
        ("B", "R1", "CITES", {}),
        ("B", "R3", "CITES", {}),
        ("C", "R4", "CITES", {}),
    ]
    repo.get_all_edges.return_value = edges
    repo.get_chunk_reference_mentions.return_value = []
    
    service = BibliographicProjectionService(repo)
    service.rebuild_projection()
    
    # Verify save_reference_corpus_stats was called with correct IDFs
    called_stats = repo.save_reference_corpus_stats.call_args[0][0]
    stats_dict = {row[0]: row[2] for row in called_stats}
    
    assert abs(stats_dict["R1"] - math.log(3 / 2)) < 1e-6
    assert abs(stats_dict["R2"] - math.log(3 / 1)) < 1e-6
    
    # Verify save_paper_reference_vectors was called with correct weights
    called_vectors = repo.save_paper_reference_vectors.call_args[0][0]
    vector_weights = {(row[0], row[1]): row[2] for row in called_vectors}
    assert abs(vector_weights[("A", "R1")] - math.log(3 / 2)) < 1e-6
    assert abs(vector_weights[("A", "R2")] - math.log(3 / 1)) < 1e-6
    
    # Verify save_edges_bulk was called with BIBLIOGRAPHIC_COUPLING edges
    bulk_edges = repo.save_edges_bulk.call_args[0][0]
    coupling_edges = [e for e in bulk_edges if e[2] == "BIBLIOGRAPHIC_COUPLING"]
    
    assert len(coupling_edges) == 2  # A -> B and B -> A
    assert coupling_edges[0][0] == "A"
    assert coupling_edges[0][1] == "B"
    props = coupling_edges[0][3]
    assert props["weight"] == 0.5  # structural_weight
    assert props["structural_weight"] == 0.5
    expected_specificity = (math.log(3/2)**2) / ((math.log(3/2)**2 + math.log(3)**2))
    assert abs(props["specificity_weight"] - expected_specificity) < 1e-6
    assert props["shared_reference_count"] == 1
    assert props["shared_reference_ids"] == ["R1"]

def test_zero_norm_case():
    # If N = 1, единственный документ цитирует R1
    # df(R1) = 1 => idf(R1) = log(1/1) = 0
    # norm = 0, no BIBLIOGRAPHIC_COUPLING edge should be created.
    repo = MagicMock()
    paper_A = Paper(id="A", title="Paper A")
    repo.get_non_placeholder_papers.return_value = [paper_A]
    edges = [("A", "R1", "CITES", {})]
    repo.get_all_edges.return_value = edges
    repo.get_chunk_reference_mentions.return_value = []
    
    service = BibliographicProjectionService(repo)
    service.rebuild_projection()
    
    # Verify stats saved is log(1/1) = 0
    called_stats = repo.save_reference_corpus_stats.call_args[0][0]
    assert called_stats[0][2] == 0.0
    
    # Verify save_edges_bulk is not called for coupling
    if repo.save_edges_bulk.called:
        bulk_edges = repo.save_edges_bulk.call_args[0][0]
        coupling_edges = [e for e in bulk_edges if e[2] == "BIBLIOGRAPHIC_COUPLING"]
        assert len(coupling_edges) == 0

# 13.4 & 13.5. Integration tests: graph writes and local projection
@pytest.mark.asyncio
async def test_integration_graph_writes_and_projection(indexer, graph_repo, vector_repo):
    # Setup files and index
    # We will simulate indexing 3 papers
    # Paper A quotes "doi:10.1234/r1" and "doi:10.1234/r2"
    # Paper B quotes "doi:10.1234/r1" and "doi:10.1234/r3"
    # Paper C quotes "doi:10.1234/r4"
    
    # Wait, let's create actual Paper objects and index them using indexer._run_pipeline_async
    # mock raw references for parser
    paper_A = Paper(id="paper_a", title="Title A", doi="10.1234/a", properties={"source_type": "paper"})
    paper_B = Paper(id="paper_b", title="Title B", doi="10.1234/b", properties={"source_type": "paper"})
    paper_C = Paper(id="paper_c", title="Title C", doi="10.1234/c", properties={"source_type": "paper"})
    
    # We will pass raw references to run_pipeline_async:
    # A quotes R1, R2
    # B quotes R1, R3
    # C quotes R4
    
    # Chunk details:
    # Paper A has chunk containing citation marker [1] or context of R1
    # Paper B has chunk containing context of R1
    
    text_A = "This chunk mentions the first reference [1]. Another sentence here."
    text_B = "Here we also cite the work of R1 [1]. Some other info."
    text_C = "Nothing special here, just random text."
    
    # Index Paper A
    await indexer._run_pipeline_async(
        paper=paper_A,
        full_text=text_A,
        refs_or_links=["Check out first: doi:10.1234/r1", "Check out second: doi:10.1234/r2"],
        is_markdown=False,
        needs_enrichment=False,
        archive_fn=None
    )
    
    # Index Paper B
    await indexer._run_pipeline_async(
        paper=paper_B,
        full_text=text_B,
        refs_or_links=["Check out first: doi:10.1234/r1", "Check out third: doi:10.1234/r3"],
        is_markdown=False,
        needs_enrichment=False,
        archive_fn=None
    )
    
    # Index Paper C
    await indexer._run_pipeline_async(
        paper=paper_C,
        full_text=text_C,
        refs_or_links=["Check out fourth: doi:10.1234/r4"],
        is_markdown=False,
        needs_enrichment=False,
        archive_fn=None
    )
    
    # Let's verify database state
    # 1. ExternalWork node created
    node_r1 = graph_repo.get_node_by_id("work:doi:10.1234/r1")
    assert node_r1 is not None
    assert node_r1[0] == "ExternalWork"
    
    # 2. Paper -> CITES -> ExternalWork edge created
    neighbors_a = graph_repo.get_neighbors("paper_a")
    cites_a = [n for n in neighbors_a if n[2] == "CITES"]
    assert len(cites_a) == 2
    
    # 3. Chunk node created
    # Chunk IDs are typically "paper_id#0"
    chunk_a = graph_repo.get_node_by_id("paper_a#0")
    assert chunk_a is not None
    assert chunk_a[0] == "Chunk"
    
    # 4. Paper -> HAS_CHUNK -> Chunk edge created
    has_chunk_a = [n for n in neighbors_a if n[2] == "HAS_CHUNK"]
    assert len(has_chunk_a) == 1
    assert has_chunk_a[0][3] == "paper_a#0"
    
    # 5. Chunk -> CITES_IN_CONTEXT -> ExternalWork created
    neighbors_chunk_a = graph_repo.get_neighbors("paper_a#0")
    cites_in_context = [n for n in neighbors_chunk_a if n[2] == "CITES_IN_CONTEXT"]
    assert len(cites_in_context) == 1
    assert cites_in_context[0][3] == "work:doi:10.1234/r1"
    
    # 6. Bibliographic coupling edge created between paper_a and paper_b
    coupling_edges = [n for n in neighbors_a if n[2] == "BIBLIOGRAPHIC_COUPLING" and n[0] == "paper_a"]
    assert len(coupling_edges) == 1
    assert coupling_edges[0][3] == "paper_b"
    import json
    coupling_props = json.loads(coupling_edges[0][5])
    assert coupling_props["structural_weight"] == 0.5
    assert coupling_props["weight"] == 0.5
    
    # 7. Chunk shared reference edge created
    chunk_shared = [n for n in neighbors_chunk_a if n[2] == "RELATED_BY_SHARED_REFERENCE" and n[0] == "paper_a#0"]
    assert len(chunk_shared) == 1
    assert chunk_shared[0][3] == "paper_b#0"
    
    # 8. Idempotency check: indexing Paper A again (simulate re-indexing)
    # Reset repo caching just in case
    indexer.invalidate_concept_cache()
    
    # Run ingestion again for Paper A (we mock detect_duplicate to bypass duplicate detection error)
    indexer.detect_duplicate = MagicMock(return_value=None)
    
    await indexer._run_pipeline_async(
        paper=paper_A,
        full_text=text_A,
        refs_or_links=["Check out first: doi:10.1234/r1", "Check out second: doi:10.1234/r2"],
        is_markdown=False,
        needs_enrichment=False,
        archive_fn=None
    )
    
    # Verify no duplicates were created
    node_r1_after = graph_repo.get_node_by_id("work:doi:10.1234/r1")
    assert node_r1_after is not None
    
    # Number of paper_a neighbors shouldn't have doubled
    neighbors_a_after = graph_repo.get_neighbors("paper_a")
    cites_a_after = [n for n in neighbors_a_after if n[2] == "CITES"]
    assert len(cites_a_after) == 2

# 13.6. Migration tests
def test_migration_run(temp_db):
    # Path to fresh temporary DB is passed in temp_db. Let's initialize a basic schema first
    conn = sqlite3.connect(temp_db)
    conn.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, label TEXT, properties TEXT);")
    conn.execute("CREATE TABLE IF NOT EXISTS edges (source_id TEXT, target_id TEXT, type TEXT, properties TEXT, PRIMARY KEY(source_id, target_id, type));")
    conn.close()
    
    # Run migration on fresh DB
    success = run_migration(temp_db)
    assert success is True
    
    # Run twice to verify idempotency
    success_second = run_migration(temp_db)
    assert success_second is True
    
    # Verify tables exist
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "reference_corpus_stats" in tables
    assert "paper_reference_vector" in tables
    assert "chunk_reference_mentions" in tables
    
    # Verify index exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = {row[0] for row in cursor.fetchall()}
    assert "idx_paper_reference_vector_work" in indexes
    assert "idx_chunk_reference_mentions_work" in indexes
    conn.close()

# 13.7. Offline tests
def test_offline_constraint(temp_db, monkeypatch):
    # Block network calls
    def block_network(*args, **kwargs):
        raise RuntimeError("Forbidden Network Call!")
    
    monkeypatch.setattr(socket, "socket", block_network)
    
    # Run migration and projection logic
    # Projection service should run fully local without making any socket calls
    repo = SQLiteGraphRepository(temp_db)
    service = BibliographicProjectionService(repo)
    
    # Rebuilding projection on an empty repo should execute cleanly
    service.rebuild_projection()

# Test 1: N=2 shared-only corpus (Request 2)
def test_n2_shared_only_corpus():
    # Paper A cites X
    # Paper B cites X
    # N = 2
    # df(X) = 2 => idf(X) = ln(2/2) = 0
    # Expected: BIBLIOGRAPHIC_COUPLING(A,B) exists
    # structural_weight = 1.0
    # specificity_weight = 0.0
    # shared_reference_count = 1
    repo = MagicMock()
    paper_A = Paper(id="A", title="Paper A")
    paper_B = Paper(id="B", title="Paper B")
    repo.get_non_placeholder_papers.return_value = [paper_A, paper_B]
    edges = [
        ("A", "X", "CITES", {}),
        ("B", "X", "CITES", {}),
    ]
    repo.get_all_edges.return_value = edges
    repo.get_chunk_reference_mentions.return_value = []
    
    service = BibliographicProjectionService(repo)
    service.rebuild_projection()
    
    # Verify save_edges_bulk was called with BIBLIOGRAPHIC_COUPLING edges
    bulk_edges = repo.save_edges_bulk.call_args[0][0]
    coupling_edges = [e for e in bulk_edges if e[2] == "BIBLIOGRAPHIC_COUPLING"]
    assert len(coupling_edges) == 2  # A -> B and B -> A
    
    props = coupling_edges[0][3]
    assert props["weight"] == 1.0 # structural_weight
    assert props["structural_weight"] == 1.0
    assert props["specificity_weight"] == 0.0
    assert props["shared_reference_count"] == 1
    assert props["shared_reference_ids"] == ["X"]

# Test 2: N counts only local indexed papers (Request 3)
def test_n_counts_only_local_indexed_papers(temp_db):
    repo = SQLiteGraphRepository(temp_db)
    # Write 2 local indexed papers
    repo.save_nodes_bulk([
        ("paper_1", "Paper", {"title": "Paper 1", "is_placeholder": False}),
        ("paper_2", "Paper", {"title": "Paper 2", "is_placeholder": False}),
    ])
    # Write 10 ExternalWork nodes, some placeholders, and chunk nodes
    repo.save_nodes_bulk([
        (f"ext_{i}", "ExternalWork", {"title": f"Ext {i}", "indexed": False})
        for i in range(10)
    ])
    repo.save_nodes_bulk([
        ("placeholder_1", "Paper", {"title": "Placeholder 1", "is_placeholder": True}),
        ("chunk_1", "Chunk", {"paper_id": "paper_1"}),
        ("concept_1", "Concept", {"name": "Concept 1"}),
    ])
    
    local_papers = repo.get_non_placeholder_papers()
    assert len(local_papers) == 2
    assert {p.id for p in local_papers} == {"paper_1", "paper_2"}

# Test 3: ExternalWork -> later indexed Paper remapping (Request 4)
def test_external_work_remapping(temp_db):
    repo = SQLiteGraphRepository(temp_db)
    # Initial state: Paper A cites ExternalWork X (with id work:doi:10.1234/x)
    ext_id = "work:doi:10.1234/x"
    repo.save_nodes_bulk([
        ("paper_a", "Paper", {"title": "Paper A", "is_placeholder": False}),
        (ext_id, "ExternalWork", {
            "title": "Work X",
            "indexed": False,
            "doi": "10.1234/x",
            "canonicalization_method": "doi"
        }),
        ("paper_a#0", "Chunk", {"paper_id": "paper_a"})
    ])
    repo.save_edges_bulk([
        ("paper_a", ext_id, "CITES", {"observed": True}),
        ("paper_a#0", ext_id, "CITES_IN_CONTEXT", {"observed": True})
    ])
    # Add chunk mention
    repo.save_chunk_reference_mentions([(
        "paper_a#0", "paper_a", ext_id, "[1]", "Context text", 1, "", "doi:10.1234/x"
    )])
    
    # Later, user indexes the actual Paper X (with ID "paper_x" and doi "10.1234/x")
    repo.save_nodes_bulk([
        ("paper_x", "Paper", {"title": "Paper X", "doi": "10.1234/x", "is_placeholder": False})
    ])
    
    # Run rebuild projection
    service = BibliographicProjectionService(repo)
    service.rebuild_projection()
    
    # Verify:
    # 1. CITES edge target remapped from ext_id to paper_x
    neighbors_a = repo.get_neighbors("paper_a")
    cites = [n for n in neighbors_a if n[2] == "CITES"]
    assert len(cites) == 1
    assert cites[0][3] == "paper_x"
    
    # 2. CITES_IN_CONTEXT target remapped from ext_id to paper_x
    neighbors_chunk = repo.get_neighbors("paper_a#0")
    cites_context = [n for n in neighbors_chunk if n[2] == "CITES_IN_CONTEXT"]
    assert len(cites_context) == 1
    assert cites_context[0][3] == "paper_x"
    
    # 3. Mention work_id remapped
    mentions = repo.get_chunk_reference_mentions()
    assert len(mentions) == 1
    assert mentions[0]["work_id"] == "paper_x"
    
    # 4. ExternalWork node deleted
    assert repo.get_node_properties(ext_id) is None

# Test 4: Delete stale derived edges on reindex/rebuild (Request 5)
def test_delete_stale_derived_edges(temp_db):
    repo = SQLiteGraphRepository(temp_db)
    # Initial: A cites X, B cites X
    repo.save_nodes_bulk([
        ("A", "Paper", {"title": "Paper A", "is_placeholder": False}),
        ("B", "Paper", {"title": "Paper B", "is_placeholder": False}),
        ("X", "ExternalWork", {"title": "Work X", "indexed": False}),
        ("A#0", "Chunk", {"paper_id": "A"}),
        ("B#0", "Chunk", {"paper_id": "B"})
    ])
    # Add observed edges
    repo.save_edges_bulk([
        ("A", "X", "CITES", {}),
        ("B", "X", "CITES", {}),
        ("A#0", "X", "CITES_IN_CONTEXT", {}),
        ("B#0", "X", "CITES_IN_CONTEXT", {})
    ])
    # Add mentions
    repo.save_chunk_reference_mentions([
        ("A#0", "A", "X", "[1]", "Context A", 1, "", "X"),
        ("B#0", "B", "X", "[1]", "Context B", 1, "", "X")
    ])
    
    # Run rebuild
    service = BibliographicProjectionService(repo)
    service.rebuild_projection()
    
    # Verify BIBLIOGRAPHIC_COUPLING exists
    n_a = repo.get_neighbors("A")
    coupling = [n for n in n_a if n[2] == "BIBLIOGRAPHIC_COUPLING" and n[0] == "A"]
    assert len(coupling) == 1
    
    # Now simulate reindexing A where it no longer cites X:
    # We delete paper A's old chunk nodes, mentions and outgoing edges (simulating indexer)
    repo.delete_chunk_nodes_for_paper("A")
    repo.delete_chunk_reference_mentions_for_paper("A")
    repo.delete_edges_by_source("A", ["CITES", "HAS_CHUNK", "RELATED_TO"])
    
    # Run rebuild again
    service.rebuild_projection()
    
    # Verify BIBLIOGRAPHIC_COUPLING and RELATED_BY_SHARED_REFERENCE are removed
    n_a_after = repo.get_neighbors("A")
    coupling_after = [n for n in n_a_after if n[2] == "BIBLIOGRAPHIC_COUPLING"]
    assert len(coupling_after) == 0
    
    # Also chunk-level shared reference edges are gone
    # Chunk A#0 was deleted by delete_chunk_nodes_for_paper, but B#0 is still there.
    # B#0 should have no RELATED_BY_SHARED_REFERENCE edges.
    n_b_chunk = repo.get_neighbors("B#0")
    shared_b = [n for n in n_b_chunk if n[2] == "RELATED_BY_SHARED_REFERENCE"]
    assert len(shared_b) == 0

# Test 5: Schema parity (Request 9)
def test_schema_parity(temp_db):
    # Path for migrated DB
    migrated_db = temp_db.replace(".db", "_migrated.db")
    
    # Create migrated DB and initialize basic old tables
    # Path for migrated DB
    migrated_db = temp_db.replace(".db", "_migrated.db")
    
    # Create migrated DB and initialize basic old tables
    conn = sqlite3.connect(migrated_db)
    conn.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, label TEXT, properties TEXT);")
    conn.execute("CREATE TABLE IF NOT EXISTS edges (source_id TEXT, target_id TEXT, type TEXT, properties TEXT, PRIMARY KEY(source_id, target_id, type));")
    conn.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, text_content TEXT, page_number INTEGER, embedding BLOB, id_hash INTEGER, parent_id TEXT);")
    conn.execute("CREATE TABLE IF NOT EXISTS parent_chunks (id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, text_content TEXT, page_number INTEGER);")
    conn.close()
    
    # Run migration on migrated DB
    success = run_migration(migrated_db)
    assert success is True
    
    # Instantiate SQLiteGraphRepository on both databases to ensure they are initialized
    SQLiteGraphRepository(temp_db)
    SQLiteGraphRepository(migrated_db)
    
    # Get schemas of fresh DB (which is temp_db, auto-initialized by GraphRepository/VectorRepository in conftest)
    conn_fresh = sqlite3.connect(temp_db)
    cursor = conn_fresh.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'index') ORDER BY name")
    fresh_schema = {row[0]: row[1] for row in cursor.fetchall() if row[0] in [
        "reference_corpus_stats", "paper_reference_vector", "chunk_reference_mentions",
        "idx_paper_reference_vector_work", "idx_chunk_reference_mentions_work",
        "idx_chunk_reference_mentions_chunk", "idx_chunk_reference_mentions_paper"
    ]}
    conn_fresh.close()
    
    # Get schemas of migrated DB
    conn_mig = sqlite3.connect(migrated_db)
    cursor = conn_mig.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'index') ORDER BY name")
    mig_schema = {row[0]: row[1] for row in cursor.fetchall() if row[0] in [
        "reference_corpus_stats", "paper_reference_vector", "chunk_reference_mentions",
        "idx_paper_reference_vector_work", "idx_chunk_reference_mentions_work",
        "idx_chunk_reference_mentions_chunk", "idx_chunk_reference_mentions_paper"
    ]}
    conn_mig.close()
    
    # Clean up migrated db file
    if os.path.exists(migrated_db):
        os.remove(migrated_db)
        
    def norm_sql(s):
        return "".join(s.lower().split()) if s else ""
        
    normalized_fresh = {k: norm_sql(v) for k, v in fresh_schema.items()}
    normalized_mig = {k: norm_sql(v) for k, v in mig_schema.items()}
    assert normalized_fresh == normalized_mig

# Test 6: Directed Symmetry (Request 6)
def test_directed_symmetry():
    # Verify that BIBLIOGRAPHIC_COUPLING and RELATED_BY_SHARED_REFERENCE are written in both directions
    repo = MagicMock()
    paper_A = Paper(id="A", title="Paper A")
    paper_B = Paper(id="B", title="Paper B")
    paper_C = Paper(id="C", title="Paper C")
    repo.get_non_placeholder_papers.return_value = [paper_A, paper_B, paper_C]
    edges = [
        ("A", "X", "CITES", {}),
        ("B", "X", "CITES", {}),
    ]
    repo.get_all_edges.return_value = edges
    
    # Mock chunk mentions to check chunk-level symmetry
    repo.get_chunk_reference_mentions.return_value = [
        {"chunk_id": "A#0", "paper_id": "A", "work_id": "X", "context": "Context A"},
        {"chunk_id": "B#0", "paper_id": "B", "work_id": "X", "context": "Context B"}
    ]
    
    service = BibliographicProjectionService(repo)
    service.rebuild_projection()
    
    # Check edges written
    bulk_edges = repo.save_edges_bulk.call_args_list
    assert len(bulk_edges) == 2  # First for BIBLIOGRAPHIC_COUPLING, second for RELATED_BY_SHARED_REFERENCE
    
    coupling_edges = bulk_edges[0][0][0]
    assert len(coupling_edges) == 2
    assert {e[0] for e in coupling_edges} == {"A", "B"}
    assert {e[1] for e in coupling_edges} == {"A", "B"}
    
    chunk_edges = bulk_edges[1][0][0]
    assert len(chunk_edges) == 2
    assert {e[0] for e in chunk_edges} == {"A#0", "B#0"}
    assert {e[1] for e in chunk_edges} == {"A#0", "B#0"}

# Test 7: Chunk-level Cardinality (Request 7)
def test_chunk_level_cardinality_diagnostic(caplog):
    import logging
    # Set logging to info level
    with caplog.at_level(logging.INFO):
        repo = MagicMock()
        paper_A = Paper(id="A", title="Paper A")
        paper_B = Paper(id="B", title="Paper B")
        paper_C = Paper(id="C", title="Paper C")
        repo.get_non_placeholder_papers.return_value = [paper_A, paper_B, paper_C]
        repo.get_all_edges.return_value = [
            ("A", "X", "CITES", {}),
            ("B", "X", "CITES", {}),
        ]
        
        # Paper A has 2 chunks citing X, Paper B has 3 chunks citing X
        # Cross-paper chunk pairs: 2 * 3 = 6 unordered pairs → 12 directed edges
        repo.get_chunk_reference_mentions.return_value = [
            {"chunk_id": "A#0", "paper_id": "A", "work_id": "X"},
            {"chunk_id": "A#1", "paper_id": "A", "work_id": "X"},
            {"chunk_id": "B#0", "paper_id": "B", "work_id": "X"},
            {"chunk_id": "B#1", "paper_id": "B", "work_id": "X"},
            {"chunk_id": "B#2", "paper_id": "B", "work_id": "X"}
        ]
        
        service = BibliographicProjectionService(repo)
        service.rebuild_projection()
        
        # Verify single summary log message is generated
        # number_of_citing_chunks = 5, number_of_chunk_pairs = 6, directed = 12
        assert any(
            "Shared reference cardinality diagnostic" in record.message
            and "number_of_citing_chunks=5" in record.message
            and "number_of_chunk_pairs=6" in record.message
            and "number_of_created_chunk_edges=12" in record.message
            for record in caplog.records
        )

# Test 7b: RELATED_BY_SHARED_REFERENCE with idf=0 (N=2, df=2) — acceptance case
def test_related_by_shared_reference_idf_zero():
    # N = 2 papers, both cite X => df(X)=2, idf(X)=ln(2/2)=0
    # Chunk A cites X, Chunk B cites X
    # Expected:
    #   RELATED_BY_SHARED_REFERENCE(A#0, B#0) exists
    #   weight = structural_weight = 1.0
    #   specificity_weight = 0.0
    repo = MagicMock()
    paper_A = Paper(id="A", title="Paper A")
    paper_B = Paper(id="B", title="Paper B")
    repo.get_non_placeholder_papers.return_value = [paper_A, paper_B]
    repo.get_all_edges.return_value = [
        ("A", "X", "CITES", {}),
        ("B", "X", "CITES", {}),
    ]
    repo.get_chunk_reference_mentions.return_value = [
        {"chunk_id": "A#0", "paper_id": "A", "work_id": "X"},
        {"chunk_id": "B#0", "paper_id": "B", "work_id": "X"},
    ]

    service = BibliographicProjectionService(repo)
    service.rebuild_projection()

    # save_edges_bulk is called at least once (for BIBLIOGRAPHIC_COUPLING + chunk edges)
    assert repo.save_edges_bulk.called

    # Collect all edges across all save_edges_bulk calls
    all_edges = []
    for call in repo.save_edges_bulk.call_args_list:
        all_edges.extend(call[0][0])

    chunk_edges = [e for e in all_edges if e[2] == "RELATED_BY_SHARED_REFERENCE"]
    assert len(chunk_edges) == 2  # A#0 → B#0 and B#0 → A#0

    props = chunk_edges[0][3]
    assert props["weight"] == 1.0                # structural_weight
    assert props["structural_weight"] == 1.0
    assert props["specificity_weight"] == 0.0    # idf(X) = 0
    assert props["shared_reference_count"] == 1
    assert "X" in props["shared_reference_ids"]


# Test 8: save_edges_bulk auto-categorization fallback (Request 10)
def test_save_edges_bulk_auto_categorization(temp_db):
    repo = SQLiteGraphRepository(temp_db)
    # Ensure source paper node exists
    repo.save_nodes_bulk([
        ("paper_a", "Paper", {"title": "Paper A", "is_placeholder": False})
    ])
    
    # Save a CITES edge to target work:doi:10.1234/test (which does not exist in DB)
    repo.save_edges_bulk([
        ("paper_a", "work:doi:10.1234/test", "CITES", {})
    ])
    
    # Verify that the target node was auto-created with label ExternalWork
    properties = repo.get_node_properties("work:doi:10.1234/test")
    assert properties is not None
    # Wait, check label:
    node = repo.get_node_by_id("work:doi:10.1234/test")
    assert node[0] == "ExternalWork"
    
    # Verify a normal DOI ID is not categorized as ExternalWork (falls back to Paper or doesn't change)
    repo.save_edges_bulk([
        ("paper_a", "doi:10.1234/local", "CITES", {})
    ])
    node_local = repo.get_node_by_id("doi:10.1234/local")
    assert node_local[0] == "Paper"
