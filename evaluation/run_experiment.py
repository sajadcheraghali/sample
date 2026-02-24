from datasets import load_dataset
from evaluation.judge import judge_relevance

def evaluate_retriever(retriever_function, name, max_samples=50):

    dataset = load_dataset("carbarcha/MemoRA", split="test")

    scores = []

    for i, sample in enumerate(dataset):

        if i >= max_samples:
            break

        query = sample["question"]
        ground_truth = sample["answer"]

        retrieved_context = retriever_function(query)

        score = judge_relevance(query, retrieved_context, ground_truth)

        scores.append(score)

        print(f"{name} | Sample {i} | Score: {score}")

    return scores