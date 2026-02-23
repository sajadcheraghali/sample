from memory.stm import STM
from memory.ltm import LTM
from llm import call_llm

class HybridRetriever:

    def __init__(self, stm: STM, ltm: LTM):
        self.stm = stm
        self.ltm = ltm

    def retrieve_and_generate(self, query, debug=False):

        # 1️⃣ گرفتن context از STM
        stm_results = self.stm.retrieve(query, k=3)

        # 2️⃣ گرفتن context از LTM (iterative)
        ltm_results = self.ltm.retrieve_ltm_iterative(query)

        # 3️⃣ ترکیب context ها
        combined_context = "\n".join(stm_results + ltm_results)

        if debug:
            print("\n==============================")
            print("STM RESULTS:")
            for r in stm_results:
                print("-", r)

            print("\nLTM RESULTS:")
            for r in ltm_results:
                print("-", r)

            print("\nCOMBINED CONTEXT:")
            print(combined_context)
            print("==============================\n")

        # 4️⃣ ساخت prompt نهایی
        final_prompt = f"""
You are a memory-augmented assistant.

Relevant Context:
{combined_context}

User Query:
{query}

Answer the query using the provided context.
"""

        # 5️⃣ تولید پاسخ
        response = call_llm(final_prompt)

        return response