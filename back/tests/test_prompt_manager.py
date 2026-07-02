import unittest
import tempfile
import os
import shutil
from src.prompts.manager import PromptManager

class TestPromptManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy prompt categories and templates
        self.extraction_dir = os.path.join(self.temp_dir, "extraction")
        os.makedirs(self.extraction_dir, exist_ok=True)
        
        self.json_dir = os.path.join(self.temp_dir, "json")
        os.makedirs(self.json_dir, exist_ok=True)
        
        # Simple prompt template
        with open(os.path.join(self.extraction_dir, "concept.txt"), "w", encoding="utf-8") as f:
            f.write("Define this concept: {{name}} or {{ name }}.")
            
        # JSON template with braces that shouldn't break
        with open(os.path.join(self.json_dir, "schema.txt"), "w", encoding="utf-8") as f:
            f.write("Schema: { \"key\": \"{{value}}\" }")
            
        self.manager = PromptManager(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_prompt_basic(self):
        rendered = self.manager.get_prompt("extraction", "concept", name="Self-Attention")
        # Ensure both forms (with and without space) are replaced
        self.assertEqual(rendered, "Define this concept: Self-Attention or Self-Attention.")

    def test_get_prompt_json_braces(self):
        # Braces inside JSON shouldn't cause key errors or formatting issues, only {{value}} should be replaced
        rendered = self.manager.get_prompt("json", "schema", value="abc")
        self.assertEqual(rendered, "Schema: { \"key\": \"abc\" }")

    def test_get_prompt_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.manager.get_prompt("extraction", "nonexistent")

    def test_caching(self):
        # Retrieve once to cache it
        self.manager.get_prompt("extraction", "concept", name="A")
        
        # Modify file directly on disk
        with open(os.path.join(self.extraction_dir, "concept.txt"), "w", encoding="utf-8") as f:
            f.write("Define: {{name}} (modified)")
            
        # Retrieve again, should hit the cache and return the old version
        rendered_2 = self.manager.get_prompt("extraction", "concept", name="A")
        self.assertEqual(rendered_2, "Define this concept: A or A.")
