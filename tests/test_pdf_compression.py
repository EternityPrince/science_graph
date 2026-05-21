import os
import tempfile
import shutil
from pathlib import Path
import fitz
import pytest

from src.config import config
from src.indexer import compress_and_save_pdf

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
