#!/usr/bin/env python3
import argparse
import shutil
import sqlite3
import time
from pathlib import Path

def run_migration(db_path: str):
    db_file = Path(db_path).resolve()
    if not db_file.exists():
        print(f"Error: Database file does not exist at {db_file}")
        return False
    
    # 1. Create a backup first
    timestamp = int(time.time())
    backup_file = db_file.with_name(f"{db_file.name}.bak.{timestamp}")
    print(f"Creating database backup at {backup_file}...")
    try:
        src_conn = sqlite3.connect(str(db_file))
        src_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        dest_conn = sqlite3.connect(str(backup_file))
        with src_conn:
            src_conn.backup(dest_conn)
        dest_conn.close()
        src_conn.close()
    except Exception as e:
        print(f"Warning: WAL-safe backup failed ({e}), falling back to direct file copy...")
        shutil.copy2(db_file, backup_file)
    
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA foreign_keys = ON;")
    
    created_tables = []
    created_indexes = []
    skipped_existing = []
    
    try:
        with conn:
            # Check existing tables
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            # Check existing indexes
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            existing_indexes = {row[0] for row in cursor.fetchall()}
            
            # 2. Table: reference_corpus_stats
            table_name = "reference_corpus_stats"
            if table_name not in existing_tables:
                conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    work_id TEXT PRIMARY KEY,
                    df INTEGER NOT NULL,
                    idf REAL NOT NULL,
                    n_local_papers INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """)
                created_tables.append(table_name)
                existing_tables.add(table_name)
            else:
                skipped_existing.append(table_name)
                
            # Table: paper_reference_vector
            table_name = "paper_reference_vector"
            if table_name not in existing_tables:
                conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    paper_id TEXT NOT NULL,
                    work_id TEXT NOT NULL,
                    weight REAL NOT NULL,
                    PRIMARY KEY (paper_id, work_id),
                    FOREIGN KEY (paper_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (work_id) REFERENCES nodes(id) ON DELETE CASCADE
                );
                """)
                created_tables.append(table_name)
                existing_tables.add(table_name)
            else:
                skipped_existing.append(table_name)
                
            # Table: chunk_reference_mentions
            table_name = "chunk_reference_mentions"
            if table_name not in existing_tables:
                conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    chunk_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    work_id TEXT NOT NULL,
                    citation_marker TEXT,
                    context TEXT,
                    page_number INTEGER,
                    section TEXT,
                    raw_reference TEXT,
                    PRIMARY KEY (chunk_id, work_id),
                    FOREIGN KEY (chunk_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (paper_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (work_id) REFERENCES nodes(id) ON DELETE CASCADE
                );
                """)
                created_tables.append(table_name)
                existing_tables.add(table_name)
            else:
                skipped_existing.append(table_name)
                
            # 3. Indexes
            indexes = []
            
            # Only define indexes on tables if those tables exist
            if "edges" in existing_tables:
                indexes.extend([
                    ("idx_edges_type_source", "edges(type, source_id)"),
                    ("idx_edges_type_target", "edges(type, target_id)"),
                    ("idx_edges_source_type_target", "edges(source_id, type, target_id)"),
                ])
            else:
                print("Note: 'edges' table does not exist yet. Skipping edges indexes for now.")
                
            if "paper_reference_vector" in existing_tables:
                indexes.append(("idx_paper_reference_vector_work", "paper_reference_vector(work_id)"))
            if "chunk_reference_mentions" in existing_tables:
                indexes.append(("idx_chunk_reference_mentions_work", "chunk_reference_mentions(work_id)"))
                
            for idx_name, idx_def in indexes:
                if idx_name not in existing_indexes:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_def};")
                    created_indexes.append(idx_name)
                else:
                    skipped_existing.append(idx_name)
                    
        print("\nMigration completed successfully!")
        print(f"Created tables: {', '.join(created_tables) if created_tables else 'None'}")
        print(f"Created indexes: {', '.join(created_indexes) if created_indexes else 'None'}")
        print(f"Skipped existing: {', '.join(skipped_existing) if skipped_existing else 'None'}")
        return True
    except Exception as e:
        print(f"\nMigration failed: {e}")
        # Try to restore backup in case of critical error
        print(f"Restoring database from backup {backup_file}...")
        conn.close()
        shutil.copy2(backup_file, db_file)
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate local bibliographic projection tables.")
    parser.add_argument("--db", required=True, help="Path to science_graph SQLite database file.")
    args = parser.parse_args()
    
    success = run_migration(args.db)
    if not success:
        exit(1)
