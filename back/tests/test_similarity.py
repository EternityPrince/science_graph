import unittest
from src.services.similarity import JaccardSimilarity

class TestJaccardSimilarity(unittest.TestCase):

    def test_get_3_shingles_empty(self):
        self.assertEqual(JaccardSimilarity.get_3_shingles(""), set())
        self.assertEqual(JaccardSimilarity.get_3_shingles("  "), set())
        # Too short for a 3-word shingle
        self.assertEqual(JaccardSimilarity.get_3_shingles("hello world"), set())

    def test_get_3_shingles_basic(self):
        text = "Hello world, this is a test!"
        # Expected shingles (lowercase, stripped punctuation):
        # 1. ("hello", "world", "this")
        # 2. ("world", "this", "is")
        # 3. ("this", "is", "a")
        # 4. ("is", "a", "test")
        expected = {
            ("hello", "world", "this"),
            ("world", "this", "is"),
            ("this", "is", "a"),
            ("is", "a", "test")
        }
        self.assertEqual(JaccardSimilarity.get_3_shingles(text), expected)

    def test_shingle_jaccard_similarity(self):
        text1 = "Hello world, this is a test"
        text2 = "Hello world, this is a test"
        self.assertAlmostEqual(JaccardSimilarity.shingle_jaccard_similarity(text1, text2), 1.0)

        # Completely different
        text3 = "Apples oranges bananas pineapple"
        self.assertAlmostEqual(JaccardSimilarity.shingle_jaccard_similarity(text1, text3), 0.0)

        # Partially matching
        # text1 shingles: 4 shingles
        # text4: "Hello world, this is another test"
        # text4 shingles:
        # ("hello", "world", "this")
        # ("world", "this", "is")
        # ("this", "is", "another")
        # ("is", "another", "test")
        # Intersection: ("hello", "world", "this"), ("world", "this", "is") -> size 2
        # Union: 6 shingles -> size 6
        # Jaccard = 2/6 = 0.333
        text4 = "Hello world, this is another test"
        self.assertAlmostEqual(JaccardSimilarity.shingle_jaccard_similarity(text1, text4), 2.0 / 6.0)

    def test_word_jaccard_similarity(self):
        text1 = "hello world test"
        text2 = "hello world test"
        self.assertAlmostEqual(JaccardSimilarity.word_jaccard_similarity(text1, text2), 1.0)

        text3 = "goodbye earth"
        self.assertAlmostEqual(JaccardSimilarity.word_jaccard_similarity(text1, text3), 0.0)

        # text1: {hello, world, test}
        # text4: hello world another
        # Intersection: {hello, world} -> size 2
        # Union: {hello, world, test, another} -> size 4
        # Jaccard = 2/4 = 0.5
        text4 = "hello world another"
        self.assertAlmostEqual(JaccardSimilarity.word_jaccard_similarity(text1, text4), 0.5)

    def test_author_jaccard_similarity(self):
        authors1 = ["Alice Smith", "Bob Jones"]
        authors2 = ["bob-jones", "alice-smith"]
        self.assertAlmostEqual(JaccardSimilarity.author_jaccard_similarity(authors1, authors2), 1.0)

        authors3 = ["Charlie Brown"]
        self.assertAlmostEqual(JaccardSimilarity.author_jaccard_similarity(authors1, authors3), 0.0)

        authors4 = ["Alice Smith", "Charlie Brown"]
        # Intersection: {"alice-smith"} -> size 1
        # Union: {"alice-smith", "bob-jones", "charlie-brown"} -> size 3
        # Jaccard = 1/3
        self.assertAlmostEqual(JaccardSimilarity.author_jaccard_similarity(authors1, authors4), 1.0 / 3.0)

    def test_deduplicate_chunks_paragraph_level_empty(self):
        self.assertEqual(JaccardSimilarity.deduplicate_chunks_paragraph_level([]), [])

    def test_deduplicate_chunks_paragraph_level_no_duplicates(self):
        chunks = [
            "This is the first paragraph in chunk one.\n\nThis is the second paragraph.",
            "This is a completely different text here in chunk two."
        ]
        result = JaccardSimilarity.deduplicate_chunks_paragraph_level(chunks)
        self.assertEqual(result, chunks)

    def test_deduplicate_chunks_paragraph_level_with_exact_duplicate(self):
        chunks = [
            "This is the first paragraph.\n\nThis is the second paragraph.",
            "This is a different paragraph.\n\nThis is the second paragraph."  # duplicate paragraph here
        ]
        expected = [
            "This is the first paragraph.\n\nThis is the second paragraph.",
            "This is a different paragraph."
        ]
        result = JaccardSimilarity.deduplicate_chunks_paragraph_level(chunks)
        self.assertEqual(result, expected)

    def test_deduplicate_chunks_paragraph_level_with_near_duplicate(self):
        # "This is the second paragraph." and "This is the second paragraph with slightly more details."
        # Shingles for P1 (second para):
        # ("this", "is", "the"), ("is", "the", "second"), ("the", "second", "paragraph") -> 3 shingles
        # Shingles for P2 (second para with slightly more details):
        # ("this", "is", "the"), ("is", "the", "second"), ("the", "second", "paragraph"),
        # ("second", "paragraph", "with"), ("paragraph", "with", "slightly"), ("with", "slightly", "more"),
        # ("slightly", "more", "details") -> 7 shingles
        # Intersection: 3 shingles
        # Union: 7 shingles
        # Jaccard: 3/7 = 0.428 (under default 0.8 threshold)
        
        # Let's test with a much more similar one:
        # P1: "This is the second paragraph in this document." (5 shingles)
        # P2: "This is the second paragraph in that document."
        # sh1:
        # ("this", "is", "the"), ("is", "the", "second"), ("the", "second", "paragraph"),
        # ("second", "paragraph", "in"), ("paragraph", "in", "this"), ("in", "this", "document") -> 6 shingles
        # sh2:
        # ("this", "is", "the"), ("is", "the", "second"), ("the", "second", "paragraph"),
        # ("second", "paragraph", "in"), ("paragraph", "in", "that"), ("in", "that", "document") -> 6 shingles
        # Intersection: 4 shingles
        # Union: 8 shingles
        # Jaccard: 4/8 = 0.5. If we set threshold to 0.45, it should match and be removed.
        chunks2 = [
            "Paragraph one.\n\nThis is the second paragraph in this document.",
            "Paragraph three.\n\nThis is the second paragraph in that document."
        ]
        # Threshold = 0.45, so duplicate is detected and removed.
        result = JaccardSimilarity.deduplicate_chunks_paragraph_level(
            chunks2, paragraph_similarity_threshold=0.45
        )
        expected = [
            "Paragraph one.\n\nThis is the second paragraph in this document.",
            "Paragraph three."
        ]
        self.assertEqual(result, expected)

    def test_deduplicate_chunks_paragraph_level_completely_removed(self):
        chunks = [
            "This is the first paragraph.\n\nThis is the second paragraph.",
            "This is the first paragraph.\n\nThis is the second paragraph."  # all duplicate paragraphs
        ]
        # Second chunk should be completely removed (empty)
        expected = [
            "This is the first paragraph.\n\nThis is the second paragraph."
        ]
        result = JaccardSimilarity.deduplicate_chunks_paragraph_level(chunks)
        self.assertEqual(result, expected)
