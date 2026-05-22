import os
import tempfile
import textwrap
import unittest
import datetime
from pathlib import Path
from src.parsers.md_parser import MarkdownParser

class TestMDParserDate(unittest.TestCase):
    def _write_md(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return f.name

    def test_frontmatter_created_at(self):
        path = self._write_md(textwrap.dedent("""\
            ---
            title: "Test Note"
            created_at: "2026-01-01T12:00:00"
            ---
            Content here.
            """))
        try:
            paper, _, _ = MarkdownParser().parse(path)
            self.assertEqual(paper.created_at, "2026-01-01T12:00:00")
        finally:
            os.unlink(path)

    def test_frontmatter_created(self):
        path = self._write_md(textwrap.dedent("""\
            ---
            title: "Test Note"
            created: "2025-05-10"
            ---
            Content here.
            """))
        try:
            paper, _, _ = MarkdownParser().parse(path)
            self.assertEqual(paper.created_at, "2025-05-10")
        finally:
            os.unlink(path)

    def test_frontmatter_date_field(self):
        path = self._write_md(textwrap.dedent("""\
            ---
            title: "Test Note"
            date: "2024-12-25T18:30:00"
            ---
            Content here.
            """))
        try:
            paper, _, _ = MarkdownParser().parse(path)
            self.assertEqual(paper.created_at, "2024-12-25T18:30:00")
        finally:
            os.unlink(path)

    def test_fallback_filesystem_date(self):
        path = self._write_md(textwrap.dedent("""\
            # Title Without Frontmatter Date

            Some content.
            """))
        try:
            paper, _, _ = MarkdownParser().parse(path)
            self.assertIsNotNone(paper.created_at)
            # Should be a parseable timestamp (ISO format)
            dt = datetime.datetime.fromisoformat(paper.created_at)
            self.assertIsInstance(dt, datetime.datetime)
        finally:
            os.unlink(path)
