from llm import call_llm

def decide_action(query, group_summary, conversations, search_history):
    prompt = f"""
You are managing memory retrieval.

Query:
{query}

Current Group Summary:
{group_summary}

Retrieved Conversations:
{conversations}

Search History:
{search_history}

Choose one action:
- END (if enough information found)
- JUMP (if another group might be better)
- REWRITE: <new_query> (if query needs refinement)

Respond in JSON format:
{{
  "action": "...",
  "new_query": "..."
}}
"""

    response = call_llm(prompt)
    return response