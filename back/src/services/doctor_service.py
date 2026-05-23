import re
import json
import logging
from typing import List, Dict, Any, Tuple
from src.models import Author, Concept, slugify
from src.repository.base import GraphRepository, VectorRepository
from src.services.normalization_pipeline import NormalizationPipeline

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Cleans raw text values to remove technical tokens and other artifacts of LLM outputs.
    Specifically, it:
    1. Removes reasoning/thinking blocks (<think>...</think> or <thought>...</thought>).
    2. Removes unclosed thinking blocks at the end of text.
    3. Removes markdown code block wraps (```json ... ```).
    4. Strips outer wrapping quotes.
    5. Normalizes consecutive whitespaces and double line breaks.
    """
    if not text:
        return ""

    # 1. Remove closed think/thought/reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Remove unclosed think/thought/reasoning blocks at the end of text
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thought>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 3. Clean up leftover tags
    text = re.sub(r"</?(think|thought|reasoning)>", "", text, flags=re.IGNORECASE)

    # 4. Remove markdown code blocks (e.g. ```json ... ```)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|markdown|text|xml|html)?\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # 5. Remove wrapping quotes if they wrap the entire string
    if len(text) >= 2:
        for q in ['"', "'", '`']:
            if text.startswith(q) and text.endswith(q):
                text = text[1:-1].strip()
                break

    # 6. Replace multiple consecutive spaces with a single space (preserving newlines)
    lines = []
    for line in text.splitlines():
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)

    # 7. Join lines and replace multiple consecutive blank lines with a single blank line
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


class DoctorService:
    """
    Service to scan and sanitize all entities in the database.
    Checks and fixes unapplied formatters and LLM artifacts.
    """

    def __init__(self, graph_repo: GraphRepository, vector_repo: VectorRepository):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.normalizer = NormalizationPipeline()

    def run_diagnostics(self, fix: bool = False) -> Dict[str, Any]:
        """
        Runs diagnostics and returns a dictionary detailing changes and stats.
        If fix=True, actually writes changes to the repository.
        """
        report = {
            "stats": {
                "papers_checked": 0,
                "papers_fixed": 0,
                "concepts_checked": 0,
                "concepts_fixed": 0,
                "concepts_migrated": 0,
                "concepts_merged": 0,
                "authors_checked": 0,
                "authors_fixed": 0,
                "authors_migrated": 0,
                "authors_merged": 0,
                "chunks_checked": 0,
                "chunks_fixed": 0,
            },
            "anomalies": {
                "papers": [],
                "concepts": [],
                "authors": [],
                "chunks": [],
            }
        }

        # 1. Fetch all nodes and classify
        all_nodes = self.graph_repo.get_all_nodes()
        papers_data = []
        concepts_data = []
        authors_data = []

        for node_id, label, properties_json in all_nodes:
            try:
                props = json.loads(properties_json) if properties_json else {}
            except Exception as e:
                logger.error(f"Error parsing properties JSON for node {node_id}: {e}")
                props = {}

            if label == "Paper":
                papers_data.append((node_id, props))
            elif label == "Concept":
                concepts_data.append((node_id, props))
            elif label == "Author":
                authors_data.append((node_id, props))

        # We will retrieve all edges to handle migrations
        all_edges = self.graph_repo.get_all_edges()

        # Helper to find edges connected to an ID
        def get_edges_for_node(node_id: str) -> List[Tuple[str, str, str, Dict[str, Any]]]:
            node_edges = []
            for src, tgt, etype, eprops_json in all_edges:
                if src == node_id or tgt == node_id:
                    try:
                        eprops = json.loads(eprops_json) if eprops_json else {}
                    except Exception:
                        eprops = {}
                    node_edges.append((src, tgt, etype, eprops))
            return node_edges

        # ── Step 1: Sanitize Authors ──────────────────────────────────────────
        report["stats"]["authors_checked"] = len(authors_data)
        for author_id, props in authors_data:
            name = props.get("name", author_id)
            normalized_name = self.normalizer.normalize_author_name(name)
            new_name = clean_text(normalized_name)

            if new_name != name:
                new_id = slugify(new_name)
                
                # Check if it requires ID migration
                if new_id != author_id:
                    # Check if destination already exists
                    dest_exists = any(node[0] == new_id for node in all_nodes)
                    if dest_exists:
                        report["stats"]["authors_merged"] += 1
                        report["anomalies"]["authors"].append({
                            "id": author_id,
                            "old_name": name,
                            "new_name": new_name,
                            "action": f"merged into existing ID '{new_id}'"
                        })
                        if fix:
                            # Migrate edges
                            edges = get_edges_for_node(author_id)
                            for src, tgt, etype, eprops in edges:
                                new_src = new_id if src == author_id else src
                                new_tgt = new_id if tgt == author_id else tgt
                                # Avoid duplicating self-loops or identical edges if they already exist
                                self.graph_repo.add_edge(new_src, new_tgt, etype, eprops)
                            self.graph_repo.delete_node(author_id)
                    else:
                        report["stats"]["authors_migrated"] += 1
                        report["anomalies"]["authors"].append({
                            "id": author_id,
                            "old_name": name,
                            "new_name": new_name,
                            "action": f"migrated to new ID '{new_id}'"
                        })
                        if fix:
                            # Create new node
                            new_props = {**props, "name": new_name}
                            new_author = Author(id=new_id, name=new_name, properties=new_props)
                            self.graph_repo.save_author(new_author)
                            
                            # Migrate edges
                            edges = get_edges_for_node(author_id)
                            for src, tgt, etype, eprops in edges:
                                new_src = new_id if src == author_id else src
                                new_tgt = new_id if tgt == author_id else tgt
                                self.graph_repo.add_edge(new_src, new_tgt, etype, eprops)
                            
                            # Delete old node
                            self.graph_repo.delete_node(author_id)
                else:
                    # Same ID, just update properties
                    report["stats"]["authors_fixed"] += 1
                    report["anomalies"]["authors"].append({
                        "id": author_id,
                        "old_name": name,
                        "new_name": new_name,
                        "action": "properties updated"
                    })
                    if fix:
                        new_props = {**props, "name": new_name}
                        self.graph_repo.update_node_properties(author_id, new_props)

        # ── Step 2: Sanitize Concepts ─────────────────────────────────────────
        report["stats"]["concepts_checked"] = len(concepts_data)
        for concept_id, props in concepts_data:
            name = props.get("name", concept_id)
            description = props.get("description", "")
            is_tag = props.get("is_tag", False)

            # Apply appropriate formatter
            if is_tag:
                normalized_name = self.normalizer.normalize_tag(name)
            else:
                normalized_name = self.normalizer.normalize_concept_name(name)

            new_name = clean_text(normalized_name)
            new_desc = clean_text(self.normalizer.normalize_description(description))

            if new_name != name or new_desc != description:
                new_id = slugify(new_name)
                
                # Check if it requires ID migration
                if new_id != concept_id:
                    # Check if destination already exists
                    dest_exists = any(node[0] == new_id for node in all_nodes)
                    if dest_exists:
                        report["stats"]["concepts_merged"] += 1
                        report["anomalies"]["concepts"].append({
                            "id": concept_id,
                            "old_name": name,
                            "new_name": new_name,
                            "old_description": description[:60] + "..." if description else "",
                            "new_description": new_desc[:60] + "..." if new_desc else "",
                            "action": f"merged into existing ID '{new_id}'"
                        })
                        if fix:
                            # Fetch existing node properties to merge description if needed
                            existing_label_props = self.graph_repo.get_node_by_id(new_id)
                            merged_desc = new_desc
                            if existing_label_props:
                                try:
                                    existing_props = json.loads(existing_label_props[1])
                                    merged_desc = existing_props.get("description") or new_desc
                                except Exception:
                                    pass
                            
                            # Update the existing node properties
                            self.graph_repo.update_node_properties(new_id, {
                                **props, 
                                "name": new_name, 
                                "description": merged_desc
                            })
                            
                            # Migrate edges
                            edges = get_edges_for_node(concept_id)
                            for src, tgt, etype, eprops in edges:
                                new_src = new_id if src == concept_id else src
                                new_tgt = new_id if tgt == concept_id else tgt
                                self.graph_repo.add_edge(new_src, new_tgt, etype, eprops)
                            
                            self.graph_repo.delete_node(concept_id)
                    else:
                        report["stats"]["concepts_migrated"] += 1
                        report["anomalies"]["concepts"].append({
                            "id": concept_id,
                            "old_name": name,
                            "new_name": new_name,
                            "old_description": description[:60] + "..." if description else "",
                            "new_description": new_desc[:60] + "..." if new_desc else "",
                            "action": f"migrated to new ID '{new_id}'"
                        })
                        if fix:
                            # Create new node
                            new_props = {**props, "name": new_name, "description": new_desc}
                            new_concept = Concept(id=new_id, name=new_name, properties=new_props)
                            self.graph_repo.save_concept(new_concept)
                            
                            # Migrate edges
                            edges = get_edges_for_node(concept_id)
                            for src, tgt, etype, eprops in edges:
                                new_src = new_id if src == concept_id else src
                                new_tgt = new_id if tgt == concept_id else tgt
                                self.graph_repo.add_edge(new_src, new_tgt, etype, eprops)
                            
                            # Delete old node
                            self.graph_repo.delete_node(concept_id)
                else:
                    # Same ID, just update properties
                    report["stats"]["concepts_fixed"] += 1
                    report["anomalies"]["concepts"].append({
                        "id": concept_id,
                        "old_name": name,
                        "new_name": new_name,
                        "old_description": description[:60] + "..." if description else "",
                        "new_description": new_desc[:60] + "..." if new_desc else "",
                        "action": "properties updated"
                    })
                    if fix:
                        new_props = {**props, "name": new_name, "description": new_desc}
                        self.graph_repo.update_node_properties(concept_id, new_props)

        # ── Step 3: Sanitize Papers ───────────────────────────────────────────
        report["stats"]["papers_checked"] = len(papers_data)
        for paper_id, props in papers_data:
            title = props.get("title", "")
            abstract = props.get("abstract", "")
            authors = props.get("authors", [])

            new_title = clean_text(title)
            new_abstract = clean_text(abstract)
            
            # Clean and normalize author list inside paper properties
            new_authors = []
            for author in authors:
                norm_author = self.normalizer.normalize_author_name(author)
                new_authors.append(clean_text(norm_author))

            if new_title != title or new_abstract != abstract or new_authors != authors:
                report["stats"]["papers_fixed"] += 1
                report["anomalies"]["papers"].append({
                    "id": paper_id,
                    "old_title": title,
                    "new_title": new_title,
                    "old_abstract": abstract[:60] + "..." if abstract else "",
                    "new_abstract": new_abstract[:60] + "..." if new_abstract else "",
                    "old_authors": authors,
                    "new_authors": new_authors,
                })
                if fix:
                    new_props = {
                        **props,
                        "title": new_title,
                        "abstract": new_abstract,
                        "authors": new_authors
                    }
                    self.graph_repo.update_node_properties(paper_id, new_props)

        # ── Step 4: Sanitize Chunks ───────────────────────────────────────────
        try:
            all_chunks = self.vector_repo.get_all_chunks()
        except Exception as e:
            logger.error(f"Error fetching chunks from vector repository: {e}")
            all_chunks = []

        report["stats"]["chunks_checked"] = len(all_chunks)
        chunks_to_save = []
        for chunk in all_chunks:
            text_content = chunk.text_content
            new_text = clean_text(text_content)
            if new_text != text_content:
                report["stats"]["chunks_fixed"] += 1
                report["anomalies"]["chunks"].append({
                    "id": chunk.id,
                    "paper_id": chunk.paper_id,
                    "old_text": text_content[:100] + "...",
                    "new_text": new_text[:100] + "..."
                })
                if fix:
                    chunk.text_content = new_text
                    chunks_to_save.append(chunk)

        if fix and chunks_to_save:
            # Save chunks in batches of 100
            for i in range(0, len(chunks_to_save), 100):
                self.vector_repo.save_chunks(chunks_to_save[i:i+100])

        return report
