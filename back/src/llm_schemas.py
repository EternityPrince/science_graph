import re
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field, RootModel

class LLMConcept(BaseModel):
    name: str
    description: str

class LLMExtractionResponse(BaseModel):
    authors: List[str] = Field(default_factory=list)
    concepts: List[LLMConcept] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class LLMClusteringResponse(RootModel[Dict[str, List[str]]]):
    pass

def validate_extraction_response(raw_data: dict) -> Tuple[LLMExtractionResponse, List[str]]:
    """
    Validates LLM extraction response at structured level,
    filtering out noisy data and providing a detailed list of warnings.
    """
    warnings = []
    
    authors_raw = raw_data.get("authors", [])
    concepts_raw = raw_data.get("concepts", [])
    tags_raw = raw_data.get("tags", [])
    
    if not isinstance(authors_raw, list):
        warnings.append("Expected 'authors' to be a list, resetting to empty.")
        authors_raw = []
    if not isinstance(concepts_raw, list):
        warnings.append("Expected 'concepts' to be a list, resetting to empty.")
        concepts_raw = []
    if not isinstance(tags_raw, list):
        warnings.append("Expected 'tags' to be a list, resetting to empty.")
        tags_raw = []

    # 1. Validate authors
    valid_authors = []
    for author in authors_raw:
        if not isinstance(author, str):
            warnings.append(f"Ignored non-string author: {author}")
            continue
        cleaned = author.strip()
        if not cleaned:
            continue
            
        lower_auth = cleaned.lower()
        bad_patterns = [
            "university", "dept", "department", "school", "college", "institute", 
            "laboratory", "labs", "research", "academy", "corporation", "inc.", 
            "co.", "ltd.", "et al", "http", "www", "journal", "proceeding", 
            "vol.", "no.", "pp.", "email", "correspondence", "author", "published"
        ]
        if "@" in lower_auth or any(pat in lower_auth for pat in bad_patterns):
            warnings.append(f"Filtered out institutional/noisy author entry: '{cleaned}'")
            continue
            
        cleaned = re.sub(r"\d+", "", cleaned)
        cleaned = re.sub(r"[.,;]+$", "", cleaned)
        cleaned = re.sub(r'^["\']|["\']$', "", cleaned)
        cleaned = cleaned.strip()
        
        if len(cleaned) < 2 or not re.search(r'[A-Za-z\u0400-\u04FF]', cleaned):
            warnings.append(f"Filtered out invalid author name format: '{author}'")
            continue
            
        valid_authors.append(cleaned)

    # 2. Validate concepts
    valid_concepts = []
    for concept in concepts_raw:
        if not isinstance(concept, dict):
            warnings.append(f"Ignored non-dict concept item: {concept}")
            continue
        c_name = concept.get("name", "")
        c_desc = concept.get("description", "")
        
        if not isinstance(c_name, str) or not isinstance(c_desc, str):
            warnings.append(f"Ignored concept with non-string name/description: {concept}")
            continue
            
        c_name = c_name.strip()
        c_desc = c_desc.strip()
        
        if not c_name:
            warnings.append("Filtered out concept with empty name.")
            continue
            
        # Clean concept name from citation brackets
        c_name = re.sub(r"\[\d+\]", "", c_name)
        c_name = re.sub(r"\(\s*[\w\s,]+,\s*\d{4}\s*\)", "", c_name)
        c_name = re.sub(r'^["\']|["\']$', "", c_name).strip()
        
        word_count = len(c_name.split())
        if word_count > 5:
            warnings.append(f"Filtered out concept '{c_name}' because the name is too long ({word_count} words). Expected 1-3 words.")
            continue
            
        desc_word_count = len(c_desc.split())
        if c_desc and (desc_word_count < 3 or desc_word_count > 60):
            warnings.append(f"Warning: Concept '{c_name}' description has unusual length ({desc_word_count} words).")
            
        valid_concepts.append(LLMConcept(name=c_name, description=c_desc))

    # 3. Validate tags
    valid_tags = []
    for tag in tags_raw:
        if not isinstance(tag, str):
            warnings.append(f"Ignored non-string tag: {tag}")
            continue
        cleaned = tag.strip().lower()
        if not cleaned:
            continue
            
        word_count = len(cleaned.split())
        if word_count > 4:
            warnings.append(f"Filtered out tag '{tag}' because it is too long ({word_count} words). Expected 1-3 words.")
            continue
            
        if re.search(r"\[\d+\]", cleaned) or any(bad in cleaned for bad in ["@", "*", "http"]):
            warnings.append(f"Filtered out tag with invalid characters: '{tag}'")
            continue
            
        valid_tags.append(cleaned)

    model = LLMExtractionResponse(
        authors=valid_authors,
        concepts=valid_concepts,
        tags=valid_tags
    )
    return model, warnings

def validate_clustering_response(raw_data: dict) -> Tuple[LLMClusteringResponse, List[str]]:
    """
    Validates LLM clustering response at structured level,
    filtering out invalid structures and chunk IDs, returning list of warnings.
    """
    warnings = []
    if not isinstance(raw_data, dict):
        warnings.append("Expected clustering response to be a JSON dictionary.")
        return LLMClusteringResponse.model_validate({}), warnings
        
    cleaned_dict = {}
    for section_title, chunk_ids in raw_data.items():
        if not isinstance(section_title, str):
            warnings.append(f"Ignored non-string section title: {section_title}")
            continue
        title = section_title.strip()
        if not title:
            warnings.append("Filtered out section with empty title.")
            continue
            
        if len(title.split()) > 10:
            warnings.append(f"Warning: Section title '{title}' is unusually long.")
            
        if not isinstance(chunk_ids, list):
            warnings.append(f"Ignored section '{title}' because chunk IDs is not a list.")
            continue
            
        valid_chunk_ids = []
        for cid in chunk_ids:
            if not isinstance(cid, str):
                warnings.append(f"Ignored non-string chunk ID '{cid}' in section '{title}'.")
                continue
            cid = cid.strip()
            if not cid:
                continue
            valid_chunk_ids.append(cid)
            
        if not valid_chunk_ids:
            warnings.append(f"Filtered out section '{title}' because it has no valid chunk IDs.")
            continue
            
        cleaned_dict[title] = valid_chunk_ids
        
    model = LLMClusteringResponse.model_validate(cleaned_dict)
    return model, warnings

class LLMVerificationResponse(BaseModel):
    relevant: bool = Field(description="True if the chunk text contains relevant educational, informational, or scientific concepts/knowledge for the database. False if it is advertisement, self-promotion, sponsor content/reads, intro/outro chat, or irrelevant filler.")
    reason: str = Field(description="Brief explanation of why the chunk is relevant or irrelevant.")

class LLMVideoSummaryResponse(BaseModel):
    overview: str = Field(description="A concise summary/overview of the video (2-3 paragraphs).")
    themes: List[str] = Field(description="List of key themes or topics discussed in the video, with brief explanations.")
    outline: List[str] = Field(description="Detailed lecture outline or chronological/structured breakdown of the video's content.")

