from app import QueryRequest, ask, evaluate_answer


dataset = [{"q": "Is RAG good?", "ground_truth": "Supposed to be."}]

for item in dataset:
    response = ask(QueryRequest(question=item["q"]))
    score = evaluate_answer(item["q"], response["context"], response["answer"])
    print(score)
