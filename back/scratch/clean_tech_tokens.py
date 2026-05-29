#!/usr/bin/env python3
import sys
import sqlite3
import json
from pathlib import Path

# Add project root to python path to load src
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.config import config
from src.llm_engine.base import strip_thinking_tokens

def clean_database():
    db_path = config.db_path
    print(f"Connecting to database: {db_path}")
    
    if not Path(db_path).exists():
        print("Database does not exist.")
        sys.exit(1)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, properties FROM nodes WHERE label = 'Paper'")
    rows = cursor.fetchall()
    
    updated_count = 0
    
    for paper_id, properties_json in rows:
        try:
            props = json.loads(properties_json)
        except Exception as e:
            print(f"Failed to parse JSON for paper {paper_id}: {e}")
            continue
            
        dirty = False
        
        # Clean summary if exists
        summary = props.get("summary")
        if summary and isinstance(summary, str):
            cleaned_summary = strip_thinking_tokens(summary)
            if cleaned_summary != summary:
                props["summary"] = cleaned_summary
                dirty = True
                
        # Clean abstract if exists
        abstract = props.get("abstract")
        if abstract and isinstance(abstract, str):
            cleaned_abstract = strip_thinking_tokens(abstract)
            if cleaned_abstract != abstract:
                props["abstract"] = cleaned_abstract
                dirty = True
                
        # Clean video properties if video_overview exists
        video_overview = props.get("video_overview")
        if video_overview and isinstance(video_overview, str):
            cleaned_overview = strip_thinking_tokens(video_overview)
            if cleaned_overview != video_overview:
                props["video_overview"] = cleaned_overview
                dirty = True

        video_themes = props.get("video_themes")
        if video_themes and isinstance(video_themes, list):
            cleaned_themes = []
            for theme in video_themes:
                if isinstance(theme, str):
                    cleaned_t = strip_thinking_tokens(theme)
                    if cleaned_t != theme:
                        dirty = True
                    cleaned_themes.append(cleaned_t)
                else:
                    cleaned_themes.append(theme)
            if dirty:
                props["video_themes"] = cleaned_themes

        video_outline = props.get("video_outline")
        if video_outline and isinstance(video_outline, list):
            cleaned_outline = []
            for outline in video_outline:
                if isinstance(outline, str):
                    cleaned_o = strip_thinking_tokens(outline)
                    if cleaned_o != outline:
                        dirty = True
                    cleaned_outline.append(cleaned_o)
                else:
                    cleaned_outline.append(outline)
            if dirty:
                props["video_outline"] = cleaned_outline
                
        if dirty:
            new_json = json.dumps(props, ensure_ascii=False)
            cursor.execute("UPDATE nodes SET properties = ? WHERE id = ?", (new_json, paper_id))
            updated_count += 1
            print(f"Cleaned technical tokens for paper: {paper_id}")
            
    if updated_count > 0:
        conn.commit()
        print(f"Successfully cleaned {updated_count} papers in the database.")
    else:
        print("No papers contained technical tokens in the database.")
        
    conn.close()

if __name__ == "__main__":
    clean_database()
