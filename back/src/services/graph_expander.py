import json
import threading
from typing import List, Tuple, Dict, Any, Optional, Set
from src.models import Chunk, Paper, Author, Concept
from src.repository.base import GraphRepository, VectorRepository
from src.llm_engine import BaseLLMEngine
from src.llm_schemas import EvidenceListResponse, EvidenceItem
from src import console as con

class ExperimentalGraphExpander:
    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        llm_engine: BaseLLMEngine,
        reranker: Any,
        p_base: float = 0.75,
        gamma: float = 0.5,
        limit: int = 5,
        top_chunks_per_paper: int = 2,
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.llm_engine = llm_engine
        self.reranker = reranker
        self.p_base = p_base
        self.gamma = gamma
        self.limit = limit
        self.top_chunks_per_paper = top_chunks_per_paper
        
        # Semaphore/lock to prevent concurrent model execution and minimize model switching overhead
        self.model_lock = threading.Lock()

    def _resolve_paper_title(self, paper_id: str, papers_map: Dict[str, Paper]) -> str:
        paper = papers_map.get(paper_id)
        if paper and paper.title:
            return paper.title
        return paper_id

    def expand(self, query: str, initial_chunks: List[Tuple[Chunk, float]]) -> str:
        """
        Runs the adaptive context expansion algorithm:
        1. Prep: extract paper IDs from initial chunks.
        2. Discovery Loop: crawl neighbors with geometric decay stopping criterion.
        3. Ingestion: load relevant chunks for discovered papers.
        4. Evidence List: filter all collected facts using LLM with short indexes.
        5. Form Prompt Block: format the essential facts as markdown text.
        """
        # Step 1: Preparation (Level 0)
        # Extract initial paper IDs
        initial_paper_ids = list({chunk.paper_id for chunk, _ in initial_chunks if chunk.paper_id})
        
        # Pull papers from DB to resolve titles and keep their details
        papers_map = self.graph_repo.get_papers_batch(initial_paper_ids)
        
        current_node_ids = set(initial_paper_ids)
        visited_nodes = set(current_node_ids)
        
        # node_id -> (label, name_or_title, card_text, connection_desc, score)
        accumulated_nodes: Dict[str, Tuple[str, str, str, Optional[str], float]] = {}
        node_labels: Dict[str, str] = {pid: "Paper" for pid in initial_paper_ids}
        
        # Populate initial papers into accumulated_nodes
        for pid in initial_paper_ids:
            paper = papers_map.get(pid)
            if paper:
                desc = f"Title: {paper.title}"
                if paper.abstract:
                    desc += f". Abstract: {paper.abstract}"
                summary = paper.properties.get("summary")
                if summary:
                    desc += f". Summary: {summary}"
                accumulated_nodes[pid] = ("Paper", paper.title, desc, "Initial search result", 1.0)

        # Step 2: Discovery Loop
        n = 1
        while n < self.limit:
            # 2.1 Bidirectional Batch Fetch
            new_neighbors: Dict[str, Tuple[str, Optional[str]]] = {}  # neighbor_id -> (label, connection_desc)
            
            for node_id in current_node_ids:
                curr_label = node_labels.get(node_id, "Paper")
                curr_title = ""
                if curr_label == "Paper":
                    paper = papers_map.get(node_id) or self.graph_repo.get_paper(node_id)
                    curr_title = paper.title if paper else node_id
                elif curr_label == "Author":
                    author = self.graph_repo.get_author(node_id)
                    curr_title = author.name if author else node_id
                elif curr_label == "Concept":
                    concept = self.graph_repo.get_concept(node_id)
                    curr_title = concept.name if concept else node_id
                
                neighbors = self.graph_repo.get_neighbors(node_id, max_depth=1)
                for src_id, src_label, edge_type, tgt_id, tgt_label, edge_props_json in neighbors:
                    # Identify neighbor
                    if src_id == node_id:
                        neigh_id = tgt_id
                        neigh_label = tgt_label
                        is_outbound = True
                    else:
                        neigh_id = src_id
                        neigh_label = src_label
                        is_outbound = False
                        
                    if neigh_id in visited_nodes or neigh_id in new_neighbors:
                        continue
                        
                    # Filter based on type:
                    # Papers can link to Paper, Author, Concept.
                    # Author and Concept can only link back to Paper.
                    if curr_label == "Paper":
                        if neigh_label not in ("Paper", "Author", "Concept"):
                            continue
                    else:
                        if neigh_label != "Paper":
                            continue
                            
                    # Construct connection description
                    conn_desc = None
                    if curr_label == "Paper":
                        if neigh_label == "Author":
                            conn_desc = f"Author of paper '{curr_title}'"
                        elif neigh_label == "Concept":
                            conn_desc = f"Concept mentioned in paper '{curr_title}'"
                        elif neigh_label == "Paper":
                            if is_outbound:
                                conn_desc = f"Cited by paper '{curr_title}'"
                            else:
                                conn_desc = f"Cites paper '{curr_title}'"
                    elif curr_label == "Author":
                        if neigh_label == "Paper":
                            conn_desc = f"Paper authored by '{curr_title}'"
                    elif curr_label == "Concept":
                        if neigh_label == "Paper":
                            conn_desc = f"Paper mentioning concept '{curr_title}'"
                            
                    new_neighbors[neigh_id] = (neigh_label, conn_desc)
                    
            if not new_neighbors:
                break
                
            # Decay strategy stopping criteria
            # K_n = len(new_neighbors) * (P_base * gamma^n)
            k_n_float = len(new_neighbors) * (self.p_base * (self.gamma ** n))
            
            # Log current hop info
            con.info(f"Hop {n}: {len(new_neighbors)} neighbors -> calculated K_n limit: {k_n_float:.2f}")
            
            if k_n_float < 1.0:
                con.info(f"Decay stopping criteria reached: K_n={k_n_float:.2f} < 1.0. Stopping crawl.")
                break
                
            k_limit = int(k_n_float)
            
            # 2.2 Summary-First Evaluation: build cards without full chunks
            cards: List[Tuple[str, str, Tuple[str, str, Optional[str]]]] = []  # List of (neigh_id, card_text, (label, name_or_title, conn_desc))
            
            # Group paper fetches for efficiency
            paper_neigh_ids = [nid for nid, (label, _) in new_neighbors.items() if label == "Paper"]
            fetched_papers = self.graph_repo.get_papers_batch(paper_neigh_ids) if paper_neigh_ids else {}
            
            for nid, (label, conn_desc) in new_neighbors.items():
                card_text = ""
                name_or_title = nid
                if label == "Paper":
                    paper = fetched_papers.get(nid) or self.graph_repo.get_paper(nid)
                    if paper:
                        name_or_title = paper.title
                        card_text = f"Title: {paper.title}"
                        if paper.abstract:
                            card_text += f". Abstract: {paper.abstract}"
                        summary = paper.properties.get("summary")
                        if summary:
                            card_text += f". Summary: {summary}"
                elif label == "Author":
                    author = self.graph_repo.get_author(nid)
                    if author:
                        name_or_title = author.name
                        # Get key works
                        papers = self.graph_repo.get_papers_by_author(nid)
                        key_works = ", ".join([p.title for p in papers if p.title])
                        card_text = f"Author Name: {author.name}. Key Works: {key_works}."
                elif label == "Concept":
                    concept = self.graph_repo.get_concept(nid)
                    if concept:
                        name_or_title = concept.name
                        description = concept.properties.get("description") or f"No description available for '{concept.name}'."
                        card_text = f"Concept Name: {concept.name}. Description: {description}."
                        
                if card_text:
                    node_labels[nid] = label
                    cards.append((nid, card_text, (label, name_or_title, conn_desc)))
                    
            if not cards:
                break
                
            # 2.3 Batch Reranking under model lock
            pairs = [(query, ctext) for _, ctext, _ in cards]
            with self.model_lock:
                scores = self.reranker.predict(pairs)
                
            # Associate scores
            scored_candidates = []
            for i, (nid, _, (label, name_or_title, conn_desc)) in enumerate(cards):
                scored_candidates.append((nid, label, name_or_title, cards[i][1], conn_desc, float(scores[i])))
                
            # Filter score > 0.4 and sort
            valid_candidates = [c for c in scored_candidates if c[5] > 0.4]
            valid_candidates.sort(key=lambda x: x[5], reverse=True)
            
            # Select top k_limit
            selected_candidates = valid_candidates[:k_limit]
            
            con.success(f"Hop {n}: selected {len(selected_candidates)} / {len(cards)} neighbors (limit {k_limit})")
            
            if not selected_candidates:
                break
                
            # Update current nodes and visited
            current_node_ids = set()
            for nid, label, name_or_title, ctext, conn_desc, score in selected_candidates:
                accumulated_nodes[nid] = (label, name_or_title, ctext, conn_desc, score)
                visited_nodes.add(nid)
                current_node_ids.add(nid)
                
            n += 1

        # Step 3: Chunk Ingestion
        # Get all surviving papers
        surviving_papers = [nid for nid, (label, _, _, _, _) in accumulated_nodes.items() if label == "Paper"]
        
        # Track already loaded chunks to avoid duplication
        loaded_chunk_ids = {c.id for c, _ in initial_chunks}
        all_chunks: List[Tuple[str, str, Optional[str], Optional[int]]] = []  # List of (chunk_id, text, paper_title, page)
        
        # Load initial chunks
        for chunk, _ in initial_chunks:
            paper_title = self._resolve_paper_title(chunk.paper_id, papers_map)
            all_chunks.append((chunk.id, chunk.text_content, paper_title, chunk.page_number))
            
        # For surviving papers, fetch additional relevant chunks if they are new
        new_chunks_to_score: List[Chunk] = []
        for pid in surviving_papers:
            paper_chunks = self.vector_repo.get_chunks_for_paper(pid)
            for c in paper_chunks:
                if c.id not in loaded_chunk_ids:
                    new_chunks_to_score.append(c)
                    
        if new_chunks_to_score:
            # Batch rerank all new chunks to find the most relevant ones
            chunk_pairs = [(query, c.text_content) for c in new_chunks_to_score]
            with self.model_lock:
                chunk_scores = self.reranker.predict(chunk_pairs)
                
            # Group chunks by paper
            paper_to_scored_chunks: Dict[str, List[Tuple[Chunk, float]]] = {}
            for idx, c in enumerate(new_chunks_to_score):
                paper_to_scored_chunks.setdefault(c.paper_id, []).append((c, float(chunk_scores[idx])))
                
            # Keep top chunks per paper
            for pid, scored in paper_to_scored_chunks.items():
                scored.sort(key=lambda x: x[1], reverse=True)
                top_scored = scored[:self.top_chunks_per_paper]
                
                # Fetch paper details to resolve title
                p_obj = self.graph_repo.get_paper(pid)
                p_title = p_obj.title if p_obj else pid
                
                for c, _ in top_scored:
                    all_chunks.append((c.id, c.text_content, p_title, c.page_number))
                    loaded_chunk_ids.add(c.id)

        # Step 4: Semantic Filtering (Evidence List)
        # Combine accumulated nodes and chunks into a single list
        # Map original IDs to short IDs (fact_1, fact_2, ...)
        orig_to_short: Dict[str, str] = {}
        short_to_orig: Dict[str, str] = {}
        short_to_text: Dict[str, str] = {}
        short_to_metadata: Dict[str, Dict[str, Any]] = {}
        
        fact_idx = 1
        
        # Process graph nodes
        for nid, (label, name_or_title, card_text, conn_desc, _) in accumulated_nodes.items():
            short_id = f"fact_{fact_idx}"
            orig_to_short[nid] = short_id
            short_to_orig[short_id] = nid
            
            # Format text card for LLM evaluation
            conn_str = f" (Connection: {conn_desc})" if conn_desc else ""
            node_desc = f"[{label}] {name_or_title}: {card_text}{conn_str}"
            short_to_text[short_id] = node_desc
            short_to_metadata[short_id] = {
                "is_node": True,
                "label": label,
                "name_or_title": name_or_title,
                "card_text": card_text,
                "conn_desc": conn_desc,
            }
            fact_idx += 1
            
        # Process chunks
        for cid, text, p_title, page in all_chunks:
            short_id = f"fact_{fact_idx}"
            orig_to_short[cid] = short_id
            short_to_orig[short_id] = cid
            
            chunk_desc = f"[Chunk] Source Paper: {p_title} (Page {page}): {text.strip()}"
            short_to_text[short_id] = chunk_desc
            short_to_metadata[short_id] = {
                "is_node": False,
                "paper_title": p_title,
                "page": page,
                "text": text,
            }
            fact_idx += 1

        if not short_to_text:
            return "No enrichment facts gathered."

        # Final request to main LLM
        prompt = (
            f"You are a strict scientific reviewer. Evaluate the following pieces of knowledge (facts) gathered from the research graph and paper chunks.\n"
            f"Determine which facts are essential/critical to answer the user query.\n\n"
            f"User Query: {query}\n\n"
            f"List of facts:\n"
        )
        for sid, text in short_to_text.items():
            prompt += f"- ID: {sid}\n  Content: {text}\n\n"
            
        prompt += (
            "Evaluate each fact. You must return a valid JSON object matching the following structure:\n"
            "{\n"
            "  \"evidence_list\": [\n"
            "    {\"id\": \"fact_1\", \"score\": 0.9, \"is_essential\": true},\n"
            "    {\"id\": \"fact_2\", \"score\": 0.3, \"is_essential\": false}\n"
            "  ]\n"
            "}\n\n"
            "Only mark \"is_essential\" as true if the fact contains critical, non-trivial information directly addressing the user's question.\n"
            "Do NOT include any conversational text or markdown formatting except the raw JSON.\n"
        )
        
        essential_items = []
        try:
            with self.model_lock:
                response = self.llm_engine.generate_and_validate_json(
                    prompt=prompt,
                    schema_class=EvidenceListResponse,
                    temp=0.0
                )
            
            # Map back short IDs
            for item in response.evidence_list:
                if item.is_essential:
                    orig_id = short_to_orig.get(item.id)
                    if orig_id:
                        essential_items.append(item.id)
                        
            con.success(f"Evidence list filtering: {len(essential_items)} / {len(short_to_text)} facts marked essential.")
        except Exception as e:
            con.warning(f"Evidence List semantic filtering failed: {e}. Falling back to including all gathered facts.")
            essential_items = list(short_to_text.keys())

        # Step 5: Format output block
        # Format the selected facts as:
        # ### KNOWLEDGE GRAPH ENRICHMENT:
        # [Тип] Название: Содержание (Связь: Описание связи, если применимо)
        enrichment_lines = []
        for sid in essential_items:
            meta = short_to_metadata.get(sid)
            if not meta:
                continue
                
            if meta["is_node"]:
                label = meta["label"]
                name_or_title = meta["name_or_title"]
                card_text = meta["card_text"]
                conn_desc = meta["conn_desc"]
                
                # Format: [Тип] Название: Содержание (Связь: Описание связи, если применимо)
                conn_suffix = f" (Связь: {conn_desc})" if conn_desc else ""
                enrichment_lines.append(f"[{label}] {name_or_title}: {card_text}{conn_suffix}")
            else:
                p_title = meta["paper_title"]
                page = meta["page"]
                text = meta["text"]
                
                page_str = f" (Page {page})" if page else ""
                enrichment_lines.append(f"[Chunk] {p_title}{page_str}: {text.strip()}")
                
        if not enrichment_lines:
            return "No essential knowledge graph enrichment found."
            
        return "\n".join(enrichment_lines)
