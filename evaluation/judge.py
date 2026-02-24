from llm import call_llm

def judge_relevance(query, retrieved_context, ground_truth):

    prompt = f"""
You are evaluating memory retrieval quality.

User Question:
{query}

Retrieved Context:
{retrieved_context}

Ground Truth Answer:
{ground_truth}

Score relevance from 1 to 10.
Respond only with a number.
"""

    response = call_llm(prompt)

    try:
        score = float(response.strip())
    except:
        score = 0

    return score