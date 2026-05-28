import os
import tempfile
import shutil
import fitz

from src.config import config
from src.parsers.pdf_parser import PDFParser
compress_and_save_pdf = PDFParser.compress_and_save_pdf

def test_pdf_compression_success():
    # Create a temp PDF with a high-res image
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
        input_path = tmp_in.name
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_out:
        output_path = tmp_out.name
        
    try:
        # Create a PDF page with an image
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        # 800x800 RGB Pixmap
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 800, 800), 0)
        rect = fitz.Rect(50, 50, 350, 350)
        page.insert_image(rect, pixmap=pix)
        doc.save(input_path, garbage=4, deflate=True)
        doc.close()
        
        orig_sz = os.path.getsize(input_path)
        assert orig_sz > 0
        
        # Compress it
        compress_and_save_pdf(
            input_path=input_path,
            output_path=output_path,
            dpi_threshold=151,
            dpi_target=150,
            quality=75
        )
        
        assert os.path.exists(output_path)
        comp_sz = os.path.getsize(output_path)
        assert comp_sz > 0
        # Re-compression should work
        assert comp_sz <= orig_sz
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

def test_get_storage_stats():
    stats = config.get_storage_stats()
    assert isinstance(stats, dict)
    assert "storage_dir" in stats
    assert "total_size" in stats
    assert "extensions" in stats
    assert "sources" in stats
    assert isinstance(stats["extensions"], list)
    assert isinstance(stats["sources"], list)

def test_pdf_compression_nested_path():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
        input_path = tmp_in.name
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "nested_dir_foo", "bar_nested", "output.pdf")
    
    try:
        doc = fitz.open()
        page = doc.new_page(width=100, height=100)
        doc.save(input_path)
        doc.close()
        
        compress_and_save_pdf(
            input_path=input_path,
            output_path=output_path,
            dpi_threshold=151,
            dpi_target=150,
            quality=75
        )
        
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def test_pdf_compression_crash_handling():
    import pytest
    with pytest.raises(RuntimeError) as exc_info:
        compress_and_save_pdf(
            input_path="non_existent_file_xyz.pdf",
            output_path="some_output_path.pdf",
            dpi_threshold=151,
            dpi_target=150,
            quality=75
        )
    assert "PDF compression process crashed" in str(exc_info.value)


def test_pdf_compression_fallback_on_rewrite_error(monkeypatch):
    import fitz
    from src.parsers.pdf_parser import _compress_worker

    # Create a temp PDF with a page
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
        input_path = tmp_in.name
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        doc = fitz.open()
        page = doc.new_page(width=100, height=100)
        doc.save(input_path)
        doc.close()

        # Mock rewrite_images to raise an exception
        def mock_rewrite_images(*args, **kwargs):
            raise RuntimeError("Mocked rewrite_images failure")

        monkeypatch.setattr(fitz.Document, "rewrite_images", mock_rewrite_images)

        # Call _compress_worker directly so the mock is active
        _compress_worker(
            input_path=input_path,
            output_path=output_path,
            dpi_threshold=151,
            dpi_target=150,
            quality=75
        )

        # The output file should still exist and be valid because of fallback
        assert os.path.exists(output_path)
        with fitz.open(output_path) as out_doc:
            assert out_doc.page_count == 1
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


def test_is_pdf_valid_helper():
    from src.parsers.pdf_parser import _is_pdf_valid
    import fitz

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        path = tmp.name

    try:
        # Create a valid PDF
        doc = fitz.open()
        doc.new_page(width=100, height=100)
        doc.save(path)
        doc.close()

        # Valid page count match
        assert _is_pdf_valid(path, 1) is True
        # Page count mismatch
        assert _is_pdf_valid(path, 2) is False

        # Corrupted / invalid path
        assert _is_pdf_valid("non_existent.pdf", 1) is False
    finally:
        if os.path.exists(path):
            os.remove(path)


