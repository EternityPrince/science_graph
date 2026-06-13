from sentence_transformers import CrossEncoder

def test():
    query = "Каковы точные параметры архитектур моделей BERT_BASE и BERT_LARGE (количество слоев, размер скрытого состояния, число голов внимания, общее число параметров), и по какой причине размер BERT_BASE был сделан именно таким?"
    
    candidates = [
        # Winograd chunk (Russian, irrelevant to BERT)
        "(9) В процессе вейвлет-обработки изображения с использованием метода Винограда K (a,q,s) входные данные разделяются на четные и нечетные группы отсчетов. При этом вычисления разбиваются на вычислительные каналы, соответствующие четным и нечетным отсчетам пикселей.",
        # BERT chunk (English, highly relevant to BERT)
        "We primarily report results on two model sizes: BERTBASE (L=12, H=768, A=12, Total Parameters=110M) and BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M). BERTBASE was chosen to have the same model size as OpenAI GPT for comparison purposes."
    ]
    
    pairs = [(query, c) for c in candidates]
    
    for model_name in [
        "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    ]:
        print(f"\nLoading {model_name}...")
        try:
            model = CrossEncoder(model_name)
            scores = model.predict(pairs)
            print(f"Scores for {model_name}:")
            print(f"  Winograd (irrelevant): {scores[0]:.4f}")
            print(f"  BERT (highly relevant): {scores[1]:.4f}")
        except Exception as e:
            print(f"Error for {model_name}: {e}")

if __name__ == "__main__":
    test()
