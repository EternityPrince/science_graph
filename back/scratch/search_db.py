import sqlite3
import json
import sys
from pathlib import Path

# Add project root to python path to load src
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.config import config

conn = sqlite3.connect(config.db_path)
cursor = conn.cursor()
cursor.execute("SELECT id, properties FROM nodes")
rows = cursor.fetchall()
print(f"Total nodes: {len(rows)}")

found = 0
for nid, properties_json in rows:
    if "im_end" in properties_json:
        print(f"Found 'im_end' in node {nid}")
        found += 1
        props = json.loads(properties_json)
        summary = props.get("summary", "")
        print(f"Summary length: {len(summary)}")
        print(f"Summary tail: {summary[-100:]!r}")
        print("-" * 50)

print(f"Search complete. Found {found} matching nodes.")
conn.close()
