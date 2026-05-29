from unittest.mock import patch, MagicMock
from src.mcp_server import (
    get_stats, search_papers, query_rag, get_paper_details,
    index_file, index_url, get_notes, create_note,
    list_notes_resource, get_note_resource, get_paper_abstract_resource,
    summarize_paper, compare_papers, analyze_concept,
    manage_graph, update_note, search_graph
)

def test_mcp_get_stats():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        mock_repo.get_stats.return_value = {"papers": 10, "authors": 5}
        mock_get_graph.return_value = mock_repo
        
        stats = get_stats()
        assert stats["papers"] == 10
        assert stats["authors"] == 5

def test_mcp_search_papers():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        paper_mock = MagicMock()
        paper_mock.id = "paper_1"
        paper_mock.title = "Test Paper"
        paper_mock.year = 2025
        paper_mock.properties = {"source_type": "paper"}
        mock_repo.search_papers_by_title.return_value = [paper_mock]
        mock_get_graph.return_value = mock_repo
        
        results = search_papers("test")
        assert len(results) == 1
        assert results[0]["id"] == "paper_1"
        assert results[0]["title"] == "Test Paper"

def test_mcp_query_rag():
    with patch("src.mcp_server.get_rag_service") as mock_get_rag:
        mock_service = MagicMock()
        mock_service.ask.return_value = "Answer context"
        mock_get_rag.return_value = mock_service
        
        res = query_rag("question", limit=10, use_cloud=True, paper_id="doc1", filters={"year_start": 2020})
        assert res == "Answer context"
        mock_service.ask.assert_called_once_with("question", limit=10, paper_id="doc1", filters={"year_start": 2020})

def test_mcp_get_paper_details_not_found():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        mock_repo.get_node_by_id.return_value = None
        mock_get_graph.return_value = mock_repo
        
        res = get_paper_details("paper_1")
        assert "error" in res

def test_mcp_get_paper_details_paper():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        mock_repo.get_node_by_id.return_value = ("Paper", '{"title": "Test Title"}')
        
        from src.models import Paper
        paper = Paper(id="paper_1", title="Test Title", authors=["Author A"], year=2025)
        mock_repo.get_paper.return_value = paper
        # MENTIONS_CONCEPT, HAS_TAG, AUTHORED, CITES, CITED_BY neighbors
        mock_repo.get_neighbors.return_value = [
            ("paper_1", "Paper", "MENTIONS_CONCEPT", "c1", "Concept", "{}"),
            ("paper_1", "Paper", "HAS_TAG", "t1", "Concept", "{}"),
            ("a1", "Author", "AUTHORED", "paper_1", "Paper", "{}"),
            ("paper_1", "Paper", "CITES", "paper_2", "Paper", "{}"),
            ("paper_3", "Paper", "CITES", "paper_1", "Paper", "{}"),
        ]
        mock_repo.get_node_by_id.side_effect = lambda nid: {
            "paper_1": ("Paper", '{"title": "Test Title"}'),
            "c1": ("Concept", '{"name": "Concept Name"}'),
            "t1": ("Concept", '{"name": "Tag Name", "is_tag": true}'),
        }.get(nid)
        
        paper2 = Paper(id="paper_2", title="Paper 2")
        paper3 = Paper(id="paper_3", title="Paper 3")
        mock_repo.get_papers_batch.side_effect = lambda ids: {
            frozenset(["paper_2"]): {"paper_2": paper2},
            frozenset(["paper_3"]): {"paper_3": paper3},
        }.get(frozenset(ids), {})

        mock_get_graph.return_value = mock_repo
        
        res = get_paper_details("paper_1")
        assert res["type"] == "paper"
        assert res["title"] == "Test Title"
        assert len(res["concepts"]) == 1
        assert res["concepts"][0]["name"] == "Concept Name"
        assert len(res["tags"]) == 1
        assert res["tags"][0]["name"] == "Tag Name"
        assert len(res["citations"]) == 1
        assert res["citations"][0]["title"] == "Paper 2"
        assert len(res["cited_by"]) == 1
        assert res["cited_by"][0]["title"] == "Paper 3"

def test_mcp_get_paper_details_author():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        mock_repo.get_node_by_id.return_value = ("Author", '{"name": "John Doe"}')
        from src.models import Paper
        mock_repo.get_papers_by_author.return_value = [
            Paper(id="p1", title="Title 1", properties={"source_type": "paper"})
        ]
        mock_get_graph.return_value = mock_repo
        
        res = get_paper_details("john_doe")
        assert res["type"] == "author"
        assert res["name"] == "John Doe"
        assert res["papers_count"] == 1
        assert res["papers"][0]["title"] == "Title 1"

def test_mcp_get_paper_details_concept():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        mock_repo.get_node_by_id.return_value = ("Concept", '{"name": "Transformers", "description": "Architecture"}')
        from src.models import Paper
        mock_repo.get_papers_by_entity.return_value = [
            Paper(id="p1", title="Title 1", properties={"source_type": "paper"})
        ]
        mock_get_graph.return_value = mock_repo
        
        res = get_paper_details("transformers")
        assert res["type"] == "concept"
        assert res["name"] == "Transformers"
        assert res["description"] == "Architecture"
        assert len(res["papers"]) == 1

def test_mcp_index_file_not_found():
    res = index_file("nonexistent.pdf")
    assert res["status"] == "error"
    assert "File not found" in res["message"]

def test_mcp_index_file_unsupported():
    with patch("src.mcp_server.os.path.exists") as mock_exists:
        mock_exists.return_value = True
        res = index_file("test.txt")
        assert res["status"] == "error"
        assert "supported" in res["message"]

@patch("src.mcp_server.os.path.exists")
@patch("src.mcp_server.Indexer")
def test_mcp_index_file_success(mock_indexer_cls, mock_exists):
    mock_exists.return_value = True
    mock_indexer = MagicMock()
    mock_indexer_cls.return_value = mock_indexer
    
    mock_indexer.index_pdf.return_value = "pdf_id"
    mock_indexer.index_markdown.return_value = "md_id"
    mock_indexer.index_epub.return_value = "epub_id"
    
    res_pdf = index_file("test.pdf")
    assert res_pdf["status"] == "success"
    assert res_pdf["id"] == "pdf_id"
    
    res_md = index_file("test.md")
    assert res_md["status"] == "success"
    assert res_md["id"] == "md_id"
    
    res_epub = index_file("test.epub")
    assert res_epub["status"] == "success"
    assert res_epub["id"] == "epub_id"

@patch("src.mcp_server.os.path.exists")
@patch("src.mcp_server.Indexer")
def test_mcp_index_file_duplicate_error(mock_indexer_cls, mock_exists):
    mock_exists.return_value = True
    mock_indexer = MagicMock()
    mock_indexer_cls.return_value = mock_indexer
    from src.indexer import DuplicateDocumentError
    mock_indexer.index_pdf.side_effect = DuplicateDocumentError("Duplicate PDF", "pdf_1")
    
    res = index_file("test.pdf")
    assert res["status"] == "error"
    assert "Duplicate document" in res["message"]

def test_mcp_index_url_empty():
    res = index_url("")
    assert res["status"] == "error"
    assert "No URLs" in res["message"]

@patch("src.mcp_server.Indexer")
@patch("src.mcp_server.get_graph_repo")
def test_mcp_index_url_success(mock_get_graph, mock_indexer_cls):
    mock_repo = MagicMock()
    mock_get_graph.return_value = mock_repo
    from src.models import Paper
    mock_repo.get_paper.return_value = Paper(id="p1", title="Paper Title")
    
    mock_indexer = MagicMock()
    mock_indexer_cls.return_value = mock_indexer
    mock_indexer.index_url.return_value = "p1"
    
    res = index_url("https://example.com")
    assert res["status"] == "success"
    assert res["id"] == "p1"
    assert res["title"] == "Paper Title"

def test_mcp_get_notes():
    with patch("src.mcp_server.get_note_service") as mock_get_note_service:
        mock_service = MagicMock()
        mock_service.get_notes.return_value = [{"title": "Note 1"}]
        mock_get_note_service.return_value = mock_service
        
        notes = get_notes()
        assert len(notes) == 1
        assert notes[0]["title"] == "Note 1"

def test_mcp_create_note_success():
    with patch("src.mcp_server.get_note_service") as mock_get_note_service:
        mock_service = MagicMock()
        mock_service.create_note.return_value = ("note_1", "path/to/note.md")
        mock_get_note_service.return_value = mock_service
        
        res = create_note("Title", "Content")
        assert res["status"] == "success"
        assert res["id"] == "note_1"
        assert res["file_path"] == "path/to/note.md"

def test_mcp_create_note_error():
    with patch("src.mcp_server.get_note_service") as mock_get_note_service:
        mock_service = MagicMock()
        mock_service.create_note.side_effect = Exception("Write error")
        mock_get_note_service.return_value = mock_service
        
        res = create_note("Title", "Content")
        assert res["status"] == "error"
        assert "Failed to create note" in res["message"]


def test_mcp_list_notes_resource():
    with patch("src.mcp_server.get_note_service") as mock_get_note_service:
        mock_service = MagicMock()
        mock_service.get_notes.return_value = [{"id": "note_1", "title": "My Note", "summary": "A summary"}]
        mock_get_note_service.return_value = mock_service
        
        res = list_notes_resource()
        assert "My Note" in res
        assert "graph://notes/note_1" in res

def test_mcp_get_note_resource():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        from src.models import Paper
        mock_repo.get_paper.return_value = Paper(id="note_1", title="Title", abstract="Abstract content", authors=["Author A"])
        mock_get_graph.return_value = mock_repo
        
        res = get_note_resource("note_1")
        assert "Abstract content" in res
        assert "Title" in res
        assert "Author A" in res

def test_mcp_get_paper_abstract_resource():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        from src.models import Paper
        mock_repo.get_paper.return_value = Paper(id="p1", title="Title", abstract="Abstract body")
        mock_get_graph.return_value = mock_repo
        
        res = get_paper_abstract_resource("p1")
        assert "Abstract body" in res

def test_mcp_prompts():
    assert "p1" in summarize_paper("p1")
    assert "p1" in compare_papers("p1", "p2")
    assert "p2" in compare_papers("p1", "p2")
    assert "c1" in analyze_concept("c1")

def test_mcp_manage_graph_delete_node():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        mock_repo.get_node_by_id.return_value = ("Paper", "{}")
        mock_get_graph.return_value = mock_repo
        
        res = manage_graph(action="delete_node", node_id="p1")
        assert res["status"] == "success"
        mock_repo.delete_node.assert_called_once_with("p1")

def test_mcp_manage_graph_create_relationship():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        mock_repo.get_node_by_id.side_effect = [("Paper", "{}"), ("Paper", "{}")]
        mock_get_graph.return_value = mock_repo
        
        res = manage_graph(action="create_edge", source_id="p1", target_id="p2", relationship_type="CITES")
        assert res["status"] == "success"
        mock_repo.add_edge.assert_called_once_with("p1", "p2", "CITES", {})

def test_mcp_manage_graph_delete_relationship():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn.execute.return_value = mock_cursor
        mock_repo._get_connection.return_value.__enter__.return_value = mock_conn
        mock_get_graph.return_value = mock_repo
        
        res = manage_graph(action="delete_edge", source_id="p1", target_id="p2", relationship_type="CITES")
        assert res["status"] == "success"
        mock_conn.execute.assert_called_once()

def test_mcp_manage_graph_add_tags():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        mock_repo.get_node_by_id.side_effect = [("Paper", "{}"), None] # paper exists, tag doesn't
        mock_get_graph.return_value = mock_repo
        
        res = manage_graph(action="add_tags", paper_id="p1", tags=["My Tag"])
        assert res["status"] == "success"
        assert "My Tag" in res["message"]
        mock_repo.save_concept.assert_called_once()
        mock_repo.add_edge.assert_called_once_with("p1", "my_tag", "HAS_TAG")

def test_mcp_update_note():
    with patch("src.mcp_server.get_note_service") as mock_get_note_service:
        mock_service = MagicMock()
        mock_service.update_note.return_value = {"status": "success", "id": "note_1", "file_path": "path.md"}
        mock_get_note_service.return_value = mock_service
        
        res = update_note(note_id="note_1", title="New Title", content="New Content", use_cloud=True)
        assert res["status"] == "success"
        assert res["id"] == "note_1"
        mock_service.update_note.assert_called_once_with(
            note_id="note_1",
            title="New Title",
            content="New Content",
            authors=None,
            tags=None
        )

def test_mcp_search_graph():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            {"id": "p1", "label": "Paper", "properties": '{"title": "Deep Learning"}', "title": "Deep Learning"},
            {"id": "a1", "label": "Author", "properties": '{"name": "Yann LeCun"}', "title": None},
            {"id": "c1", "label": "Concept", "properties": '{"name": "CNN", "is_tag": true}', "title": None},
        ]
        mock_repo._get_connection.return_value.__enter__.return_value = mock_conn
        mock_get_graph.return_value = mock_repo
        
        res = search_graph("query", limit=5)
        assert len(res) == 3
        assert res[0] == {"type": "paper", "id": "p1", "title": "Deep Learning"}
        assert res[1] == {"type": "author", "id": "a1", "name": "Yann LeCun"}
        assert res[2] == {"type": "tag", "id": "c1", "name": "CNN"}

