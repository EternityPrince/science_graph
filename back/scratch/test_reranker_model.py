from sentence_transformers import CrossEncoder

def test():
    query = "Каковы точные параметры архитектур моделей BERT_BASE и BERT_LARGE (количество слоев, размер скрытого состояния, число голов внимания, общее число параметров), и по какой причине размер BERT_BASE был сделан именно таким?"
    
    candidates = [
        # Winograd chunk
        "(9) В процессе вейвлет-обработки изображения с использованием метода Винограда K (a,q,s) входные данные разделяются на четные и нечетные группы отсчетов. При этом вычисления разбиваются на вычислительные каналы, соответствующие четным и нечетным отсчетам пикселей.",
        # BERT chunk
        "We primarily report results on two model sizes: BERTBASE (L=12, H=768, A=12, Total Parameters=110M) and BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M). BERTBASE was chosen to have the same model size as OpenAI GPT for comparison purposes."
    ]
    
    print("Loading BAAI/bge-reranker-base...")
    model = CrossEncoder("BAAI/bge-reranker-base")
    pairs = [(query, c) for c in candidates]
    scores = model.predict(pairs)
    print("Scores for bge-reranker-base:")
    print(f"  Winograd: {scores[0]}")
    print(f"  BERT: {scores[1]}")

if __name__ == "__main__":
    test()
