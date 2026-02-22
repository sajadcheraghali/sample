from llm import call_llm

def decide_action(query, group_summary, conversations, search_history):
    prompt = f"""
    You are a strict JSON generator.

    Query:
    {query}

    Group Summary:
    {group_summary}

    Retrieved Conversations:
    {conversations}

    Search History:
    {search_history}

    Respond ONLY in valid JSON.
    No explanation.
    No extra text.

    Example:
    {{
      "action": "END",
      "new_query": ""
    }}

    Or:

    {{
      "action": "REWRITE",
      "new_query": "improved query"
    }}
    """

    response = call_llm(prompt)
    return response