import os
import tempfile
import textwrap
import unittest
import datetime
from unittest.mock import patch
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

    def test_frontmatter_loads_exception(self):
        path = self._write_md("Invalid frontmatter content")
        try:
            with patch("frontmatter.loads", side_effect=Exception("Frontmatter load error")):
                paper, _, body = MarkdownParser().parse(path)
                self.assertEqual(body, "Invalid frontmatter content")
                self.assertEqual(paper.title, Path(path).stem)
        finally:
            os.unlink(path)

    def test_links_empty_and_section_only(self):
        path = self._write_md(textwrap.dedent("""\
            # Note Title
            Check out [empty_link]() and [section_link](#section-name)
            """))
        try:
            paper, links, _ = MarkdownParser().parse(path)
            self.assertEqual(links, [])
        finally:
            os.unlink(path)

    def test_filesystem_stat_exception(self):
        path = self._write_md(textwrap.dedent("""\
            # Title Without Frontmatter Date
            Content here.
            """))
        try:
            with patch("pathlib.Path.stat", side_effect=OSError("Permission error")):
                paper, _, _ = MarkdownParser().parse(path)
                self.assertIsNotNone(paper.created_at)
                dt = datetime.datetime.fromisoformat(paper.created_at)
                self.assertIsInstance(dt, datetime.datetime)
        finally:
            os.unlink(path)

    def test_frontmatter_comprehensive_types(self):
        content = textwrap.dedent("""\
            ---
            title: "Comprehensive Note"
            authors: "Author A, Author B; Author C and Author D"
            tags: "tag1, tag2"
            date: 2026-05-22
            comments_on: "comment1, comment2"
            agrees_with: "agrees1"
            disagrees_with: "disagrees1, disagrees2"
            linked_to: "link1"
            ---
            Check out inline #inline-tag.
            Also standard links:
            - [Empty]()
            - [Section Only](#section)
            - [Google](https://google.com)
            - [Relative path](subdir/my_relative_note.md#header)
            """)
        path = self._write_md(content)
        try:
            paper, links, body = MarkdownParser().parse(path)
            self.assertEqual(paper.title, "Comprehensive Note")
            self.assertEqual(paper.authors, ["Author A", "Author B", "Author C", "Author D"])
            self.assertEqual(set(paper.properties["tags"]), {"tag1", "tag2", "inline-tag"})
            self.assertEqual(paper.created_at, "2026-05-22")
            self.assertEqual(paper.properties["comments_on"], ["comment1", "comment2"])
            self.assertEqual(paper.properties["agrees_with"], ["agrees1"])
            self.assertEqual(paper.properties["disagrees_with"], ["disagrees1", "disagrees2"])
            self.assertEqual(paper.properties["linked_to"], ["link1"])
            self.assertIn("https://google.com", links)
            self.assertIn("my_relative_note", links)
        finally:
            os.unlink(path)
