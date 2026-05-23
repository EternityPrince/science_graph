#!/usr/bin/env python3
import sys
import sqlite3
from pathlib import Path

# Add project root to python path to load src
sys.path.append(str(Path(__file__).parent.resolve()))

def check_database():
    try:
        from src.config import config
    except ImportError as e:
        print(f"Error importing config: {e}")
        sys.exit(1)
        
    db_path = config.db_path
    print(f"Путь к базе данных: {db_path}")
    
    if not Path(db_path).exists():
        print("База данных не существует.")
        sys.exit(1)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("PRAGMA table_xinfo(nodes);")
        columns = [row[1] for row in cursor.fetchall()]
    except Exception as e:
        print(f"Ошибка чтения структуры таблицы nodes: {e}")
        sys.exit(1)
        
    if "is_placeholder" in columns:
        print("База данных в АКТУАЛЬНОМ состоянии (столбец is_placeholder присутствует).")
        # Показать статистику
        cursor.execute("SELECT COUNT(*) FROM nodes WHERE label = 'Paper'")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM nodes WHERE label = 'Paper' AND is_placeholder = 0")
        indexed = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM nodes WHERE label = 'Paper' AND is_placeholder = 1")
        mentioned = cursor.fetchone()[0]
        print(f"Всего работ: {total}")
        print(f"  Проиндексировано: {indexed}")
        print(f"  Упомянуто: {mentioned}")
        return True
    else:
        print("База данных ТРЕБУЕТ миграции (столбец is_placeholder отсутствует).")
        return False

def migrate_database():
    try:
        from src.config import config
    except ImportError as e:
        print(f"Error importing config: {e}")
        sys.exit(1)
        
    db_path = config.db_path
    conn = sqlite3.connect(db_path)
    try:
        print("Запуск миграции...")
        conn.execute("""
        ALTER TABLE nodes ADD COLUMN is_placeholder INTEGER GENERATED ALWAYS AS (
            CASE 
                WHEN json_extract(properties, '$.is_placeholder') = 1 THEN 1
                WHEN json_extract(properties, '$.placeholder') = 1 THEN 1
                ELSE 0
            END
        ) VIRTUAL;
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_is_placeholder ON nodes(is_placeholder);")
        conn.commit()
        print("Миграция успешно выполнена!")
    except Exception as e:
        print(f"Ошибка миграции: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if "--migrate" in sys.argv:
        migrate_database()
        check_database()
    else:
        status = check_database()
        if not status:
            print("\nЗапустите с флагом --migrate для применения миграции: python3 check_db.py --migrate")
            sys.exit(2)
        sys.exit(0)
