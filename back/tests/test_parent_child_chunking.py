import os
import tempfile
import fitz
from unittest.mock import MagicMock, patch

from src.config import config
from src.models import Chunk, Paper
from src.vector_search import split_text_to_chunks, split_segment_to_chunks
from src.services.duplicate_detector import _split_text_to_chunks_raw
from src.services.rag_service import RAGService

def test_config_parent_child_parameters():
    # Verify the config parameters exist and have correct defaults
    assert hasattr(config, "child_chunk_size")
    assert hasattr(config, "child_chunk_overlap")
    assert hasattr(config, "parent_chunk_size")
    assert hasattr(config, "parent_chunk_overlap")
    
    assert config.child_chunk_size == 300
    assert config.child_chunk_overlap == 50
    assert config.parent_chunk_size == 2500
    assert config.parent_chunk_overlap == 200

def test_split_segment_to_chunks():
    # Test split_segment_to_chunks on a simple sentence
    text = "This is a simple sentence. This is another sentence. And a third one."
    chunks = split_segment_to_chunks(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 60 # reasonable upper bound for individual sentences/groups

def test_split_text_to_chunks_parent_child():
    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = tmp.name

    try:
        doc = fitz.open()
        page = doc.new_page()
        # Insert a long text block to ensure it splits into parent and child chunks
        long_text = (
            "Sentence one of the first block of text. " * 30 +
            "\n\n" +
            "Sentence two of the second block of text. " * 30
        )
        page.insert_text((50, 50), long_text)
        doc.save(pdf_path)
        doc.close()

        # Split using split_text_to_chunks with patched small sizes
        with patch.dict(config.data["embedding"], {"parent_chunk_size": 100, "child_chunk_size": 30}):
            chunks = split_text_to_chunks(
                paper_id="test_paper",
                file_path=pdf_path
            )

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.paper_id == "test_paper"
            assert chunk.parent_id is not None
            assert chunk.parent_id.startswith("test_paper#parent_")
            assert chunk.parent_text is not None
            # The child text_content should be a substring of parent_text
            assert chunk.text_content in chunk.parent_text
            # Child chunk sizes should be within the configured child_chunk_size limits
            assert len(chunk.text_content) <= 100  # allowing some buffer for sentence reconstruction
            assert len(chunk.parent_text) > len(chunk.text_content)
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

def test_split_text_to_chunks_raw_parent_child():
    # Test raw text chunker with patched small sizes
    long_text = (
        "This is paragraph one containing some sentences. " * 20 +
        "\n\n" +
        "This is paragraph two containing some other sentences. " * 20
    )
    with patch.dict(config.data["embedding"], {"parent_chunk_size": 100, "child_chunk_size": 30}):
        chunks = _split_text_to_chunks_raw(
            paper_id="test_raw_paper",
            text=long_text
        )

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.paper_id == "test_raw_paper"
        assert chunk.parent_id is not None
        assert chunk.parent_id.startswith("test_raw_paper#parent_")
        assert chunk.parent_text is not None
        assert chunk.text_content in chunk.parent_text
        assert len(chunk.parent_text) > len(chunk.text_content)

def test_sqlite_repository_parent_child(graph_repo, vector_repo):
    # Save a chunk with parent information
    chunk = Chunk(
        id="p1#0",
        paper_id="p1",
        text_content="child text",
        page_number=1,
        embedding=[0.1] * 384,
        parent_id="p1#parent_0",
        parent_text="this is the parent text containing child text"
    )

    vector_repo.save_chunks([chunk])

    # 1. Retrieve by paper
    paper_chunks = vector_repo.get_chunks_for_paper("p1")
    assert len(paper_chunks) == 1
    retrieved = paper_chunks[0]
    assert retrieved.id == "p1#0"
    assert retrieved.parent_id == "p1#parent_0"
    assert retrieved.parent_text == "this is the parent text containing child text"

    # 2. Retrieve all
    all_chunks = vector_repo.get_all_chunks()
    assert len(all_chunks) >= 1
    match = [c for c in all_chunks if c.id == "p1#0"][0]
    assert match.parent_id == "p1#parent_0"
    assert match.parent_text == "this is the parent text containing child text"

    # 3. Retrieve via dense similarity search
    results = vector_repo.search_similar_chunks([0.1] * 384, limit=5)
    assert len(results) >= 1
    sim_chunk, score = [r for r in results if r[0].id == "p1#0"][0]
    assert sim_chunk.parent_id == "p1#parent_0"
    assert sim_chunk.parent_text == "this is the parent text containing child text"

    # 4. Retrieve via lexical FTS5 search
    fts_results = vector_repo.search_text_fts5("child", limit=5)
    assert len(fts_results) >= 1
    fts_chunk, fts_score = [r for r in fts_results if r[0].id == "p1#0"][0]
    assert fts_chunk.parent_id == "p1#parent_0"
    assert fts_chunk.parent_text == "this is the parent text containing child text"

def test_rag_build_context_uses_parent_text():
    graph_repo = MagicMock()
    vector_repo = MagicMock()
    emb_engine = MagicMock()
    llm_engine = MagicMock()
    expander = MagicMock()

    service = RAGService(graph_repo, vector_repo, emb_engine, llm_engine, expander)

    # Mock get_papers_batch
    paper1 = MagicMock(spec=Paper)
    paper1.title = "Paper Title"
    paper1.year = 2021
    paper1.authors = ["John Doe"]
    graph_repo.get_papers_batch.return_value = {"p1": paper1}

    # Chunk with parent context
    chunk = Chunk(
        id="p1#0",
        paper_id="p1",
        text_content="child text",
        page_number=1,
        parent_id="p1#parent_0",
        parent_text="This is the parent text that should be used."
    )

    # Disable graph expansion to focus on build_context output
    with patch.dict(config.data, {"rag_components": {"graph_expansion": False, "deduplicate_parent_chunks": False}}):
        context_text, context_graph = service.build_context([(chunk, 0.9)])
        
        # Verify parent text is present in the formatted context block
        assert "This is the parent text that should be used." in context_text
        # Verify child text is NOT used if parent text is present
        assert "child text" not in context_text


def test_get_parent_key_logic():
    from src.services.rag_service import _get_parent_key
    
    # 1. paper_id present
    c1 = Chunk(id="p1#0", paper_id="p1", text_content="c1", page_number=1, parent_id="p1#parent_0", parent_text="P1 Text")
    assert _get_parent_key(c1) == ("paper_id", "p1")

    # 2. No paper_id, but parent_id present
    c2 = Chunk(id="p2#0", paper_id="", text_content="short", page_number=1, parent_id="p2#parent_7", parent_text="longer parent text")
    assert _get_parent_key(c2) == ("parent_id", "p2")

    # 3. No paper_id/parent_id, but parent_text differs from text_content
    c3 = Chunk(id="p3#0", paper_id="", text_content="same text", page_number=1, parent_id=None, parent_text="different text")
    assert _get_parent_key(c3) == ("parent_text", "different text")

    # 4. No paper_id, parent_id, or distinct parent_text
    c4 = Chunk(id="p4#0", paper_id="", text_content="only text", page_number=1)
    assert _get_parent_key(c4) is None


def test_parent_chunk_deduplication_in_build_context():
    graph_repo = MagicMock()
    vector_repo = MagicMock()
    emb_engine = MagicMock()
    llm_engine = MagicMock()
    expander = MagicMock()

    service = RAGService(graph_repo, vector_repo, emb_engine, llm_engine, expander)

    p1 = MagicMock(spec=Paper)
    p1.title = "Paper 1"
    p1.year = 2021
    p1.authors = ["Author A"]

    p2 = MagicMock(spec=Paper)
    p2.title = "Paper 2"
    p2.year = 2022
    p2.authors = ["Author B"]

    graph_repo.get_papers_batch.return_value = {"p1": p1, "p2": p2}

    # 4 child chunks from Paper 1 sharing parent_id
    chunks_p1 = [
        Chunk(id=f"p1#{i}", paper_id="p1", text_content=f"c{i}", page_number=1, parent_id="p1#parent_0", parent_text="Full Abstract P1")
        for i in range(4)
    ]
    # 1 child chunk from Paper 2
    chunk_p2 = Chunk(id="p2#0", paper_id="p2", text_content="c_p2", page_number=1, parent_id="p2#parent_0", parent_text="Full Abstract P2")

    scored_chunks = [
        (chunks_p1[0], 1.747),
        (chunks_p1[1], 0.995),
        (chunks_p1[2], 0.470),
        (chunks_p1[3], -0.486),
        (chunk_p2, -4.601),
    ]

    # When deduplication is ENABLED (default)
    with patch.dict(config.data, {"rag_components": {"graph_expansion": False, "deduplicate_parent_chunks": True}}):
        context_text, _ = service.build_context(scored_chunks)
        # Should contain Block 1 and Block 2 (Paper 1 and Paper 2), but NOT Block 3, 4, 5
        assert "Block 1 (Score: 1.747)" in context_text
        assert "Block 2 (Score: -4.601)" in context_text
        assert "Block 3" not in context_text

    # When deduplication is DISABLED
    with patch.dict(config.data, {"rag_components": {"graph_expansion": False, "deduplicate_parent_chunks": False}}):
        context_text, _ = service.build_context(scored_chunks)
        # Should contain all 5 blocks
        assert "Block 1 (Score: 1.747)" in context_text
        assert "Block 2 (Score: 0.995)" in context_text
        assert "Block 3 (Score: 0.470)" in context_text
        assert "Block 4 (Score: -0.486)" in context_text
        assert "Block 5 (Score: -4.601)" in context_text


def test_parent_chunk_deduplication_in_retrieval_budget():
    graph_repo = MagicMock()
    vector_repo = MagicMock()
    emb_engine = MagicMock()
    llm_engine = MagicMock()

    service = RAGService(graph_repo, vector_repo, emb_engine, llm_engine)

    # 4 chunks for p1, 2 for p2, 1 for p3
    chunks = [
        Chunk(id="p1#0", paper_id="p1", text_content="c1_0", page_number=1, parent_id="p1#parent_0", parent_text="Text P1"),
        Chunk(id="p1#1", paper_id="p1", text_content="c1_1", page_number=1, parent_id="p1#parent_0", parent_text="Text P1"),
        Chunk(id="p1#2", paper_id="p1", text_content="c1_2", page_number=1, parent_id="p1#parent_0", parent_text="Text P1"),
        Chunk(id="p2#0", paper_id="p2", text_content="c2_0", page_number=1, parent_id="p2#parent_0", parent_text="Text P2"),
        Chunk(id="p2#1", paper_id="p2", text_content="c2_1", page_number=1, parent_id="p2#parent_0", parent_text="Text P2"),
        Chunk(id="p3#0", paper_id="p3", text_content="c3_0", page_number=1, parent_id="p3#parent_0", parent_text="Text P3"),
    ]
    
    # Mock vector repo returns
    vector_repo.search_similar_chunks.return_value = [(c, 0.9 - i*0.1) for i, c in enumerate(chunks)]
    vector_repo.search_text_fts5.return_value = []

    with patch.object(service, "expander", None):
        with patch.dict(config.data, {"rag_components": {"reranker": False, "graph_expansion": False, "dynamic_alpha_blending": False, "deduplicate_parent_chunks": True}}):
            results = service.retrieve_relevant_chunks("query", limit=3)
            
            # Should return 3 chunks corresponding to top 1 of p1, top 1 of p2, and top 1 of p3
            assert len(results) == 3
            returned_ids = [c.id for c, _ in results]
            assert returned_ids == ["p1#0", "p2#0", "p3#0"]

