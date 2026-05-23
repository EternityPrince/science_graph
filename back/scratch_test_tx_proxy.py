import sqlite3
import threading
from contextlib import contextmanager

class ConnectionProxy:
    def __init__(self, conn, is_transaction=False):
        self._conn = conn
        self._is_transaction = is_transaction

    def __enter__(self):
        if not self._is_transaction:
            self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._is_transaction:
            return self._conn.__exit__(exc_type, exc_val, exc_tb)
        return False

    def commit(self):
        if not self._is_transaction:
            self._conn.commit()

    def rollback(self):
        if not self._is_transaction:
            self._conn.rollback()

    def __getattr__(self, name):
        return getattr(self._conn, name)

class Repo:
    def __init__(self, db_path):
        self.db_path = db_path
        self._local = threading.local()
        
    def _get_connection(self):
        active_conn = getattr(self._local, "conn", None)
        if active_conn is not None:
            return ConnectionProxy(active_conn, is_transaction=True)
        conn = sqlite3.connect(self.db_path)
        return ConnectionProxy(conn, is_transaction=False)

    @contextmanager
    def transaction(self):
        if getattr(self._local, "conn", None) is not None:
            yield
            return
        conn = sqlite3.connect(self.db_path)
        self._local.conn = conn
        try:
            conn.execute("BEGIN TRANSACTION;")
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._local.conn = None
            conn.close()

    def save(self, val):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO test VALUES (?)", (val,))
            conn.commit() # This will delegate to proxy.commit()

db_path = "test_scratch2.db"
conn = sqlite3.connect(db_path)
conn.execute("CREATE TABLE IF NOT EXISTS test (id TEXT PRIMARY KEY);")
conn.commit()
conn.close()

repo = Repo(db_path)
try:
    with repo.transaction():
        repo.save("1")
        # Let's read it back using the same transaction conn
        with repo._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM test")
            print("Inside transaction:", cursor.fetchall())
        raise ValueError("Rollback please")
except ValueError:
    pass

# Test non-transactional save
repo.save("2")

conn = sqlite3.connect(db_path)
cursor = conn.execute("SELECT * FROM test")
print("After transaction (should contain only '2'):", cursor.fetchall())
conn.close()

import os
if os.path.exists(db_path):
    os.remove(db_path)
