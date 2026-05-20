import json
from typing import List, Dict, Any, Tuple
from src.models import Chunk, Paper
from src.repository.base import GraphRepository, VectorRepository
from src.vector_search import EmbeddingEngine
from src.llm_engine import LLMEngine

class RAGPipeline:
    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        llm_engine: LLMEngine
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine
        self.llm_engine = llm_engine

    def _resolve_node_name(self, node_id: str, label: str) -> str:
        """Resolves node ID to its actual name/title using DB lookup."""
        if label == "Paper":
            paper = self.graph_repo.get_paper(node_id)
            return f"'{paper.title}'" if paper else f"'{node_id}'"
        elif label == "Author":
            author = self.graph_repo.get_author(node_id)
            return author.name if author else node_id
        elif label == "Concept":
            concept = self.graph_repo.get_concept(node_id)
            return concept.name if concept else node_id
        return node_id

    def build_context(self, similar_chunks: List[tuple[Chunk, float]]) -> Tuple[str, str]:
        """
        Builds two context blocks:
        1. Semantic text blocks
        2. Knowledge graph relationships
        """
        # 1. Format text chunks
        text_blocks = []
        paper_ids = set()
        
        for idx, (chunk, score) in enumerate(similar_chunks, start=1):
            paper_ids.add(chunk.paper_id)
            paper = self.graph_repo.get_paper(chunk.paper_id)
            
            title = paper.title if paper else chunk.paper_id
            year_str = f", {paper.year}" if paper and paper.year else ""
            authors_str = f" by {', '.join(paper.authors)}" if paper and paper.authors else ""
            
            text_blocks.append(
                f"Block {idx} (Score: {score:.3f}) | Paper: {title}{authors_str}{year_str} (Page {chunk.page_number}):\n"
                f"\"\"\"\n{chunk.text_content.strip()}\n\"\"\""
            )
            
        context_text = "\n\n".join(text_blocks)

        # 2. Format Graph Subgraph around the relevant papers
        graph_lines = []
        seen_edges = set()
        
        for paper_id in paper_ids:
            neighbors = self.graph_repo.get_neighbors(paper_id, max_depth=1)
            for src_id, src_label, edge_type, tgt_id, tgt_label, edge_props in neighbors:
                edge_key = (src_id, tgt_id, edge_type)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    
                    # Resolve node display names
                    src_name = self._resolve_node_name(src_id, src_label)
                    tgt_name = self._resolve_node_name(tgt_id, tgt_label)
                    
                    if edge_type == "AUTHORED":
                        graph_lines.append(f"- {src_name} (Author) authored paper {tgt_name}")
                    elif edge_type == "MENTIONS_CONCEPT":
                        graph_lines.append(f"- Paper {src_name} mentions concept/topic '{tgt_name}'")
                    elif edge_type == "CITES":
                        try:
                            props = json.loads(edge_props) if edge_props else {}
                        except Exception:
                            props = {}
                        raw_text = props.get("raw_text")
                        if raw_text:
                            # Truncate raw reference for cleanliness
                            ref_preview = raw_text if len(raw_text) < 100 else raw_text[:100] + "..."
                            graph_lines.append(f"- Paper {src_name} cites: {ref_preview}")
                        else:
                            graph_lines.append(f"- Paper {src_name} cites paper {tgt_name}")
                    else:
                        graph_lines.append(f"- Node '{src_name}' is connected to '{tgt_name}' via {edge_type}")

        context_graph = "\n".join(graph_lines) if graph_lines else "No direct graph relations found."
        
        return context_text, context_graph

    def ask(self, query: str, limit: int = 5) -> str:
        """Runs vector search + graph retrieval and generates answers from the LLM."""
        # 1. Compute embedding of the query
        query_emb = self.emb_engine.get_embedding(query)
        
        # 2. Perform semantic search
        similar_chunks = self.vector_repo.search_similar_chunks(query_emb, limit=limit)
        if not similar_chunks:
            return "Не найдено релевантных фрагментов статей в базе данных. Пожалуйста, сначала проиндексируйте документы."

        # 3. Build context
        context_text, context_graph = self.build_context(similar_chunks)

        # 4. Construct prompt for Gemma / Qwen
        prompt = f"""<|im_start|>system
You are a research assistant. Synthesize an answer to the user's question using the retrieved text blocks and the knowledge graph connections.
Always mention the titles of the papers, years, authors, and page numbers when citation is needed.
If the graph contains citing relationships, use them to explain the context (e.g., "A cited B").

Here is the retrieved context:

### RELEVANT TEXT FRAGMENTS:
{context_text}

### KNOWLEDGE GRAPH CONNECTIONS:
{context_graph}
<|im_end|>
<|im_start|>user
Question: {query}
Answer in Russian:
<|im_end|>
<|im_start|>assistant
"""
        # 5. Generate completion
        print("[*] Generating answer using local LLM...")
        return self.llm_engine.generate_response(prompt)
