import sys
from pathlib import Path

# Add project root to python path to load src
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.container import container

def test():
    rag = container.get_rag_service(use_cloud=False)
    queries = [
        "Каковы точные параметры архитектур моделей BERT_BASE и BERT_LARGE (количество слоев, размер скрытого состояния, число голов внимания, общее число параметров), и по какой причине размер BERT_BASE был сделан именно таким?",
        "Опишите детально смешанную стратегию маскирования токенов (с указанием процентного соотношения), применяемую на случайно выбранных 15% позициях при предобучении Masked LM в BERT."
    ]
    for q in queries:
        print(f"Original: {q}")
        try:
            clean_q, filters = rag._classify_intent_and_extract_filters(q)
            print(f"Clean query: {clean_q}")
            print(f"Filters: {filters}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    test()
