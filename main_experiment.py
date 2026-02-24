from retrieval.hybrid import HybridRetriever
from memory.stm import STM
from memory.ltm import LTM

from evaluation.run_experiment import evaluate_retriever
from evaluation.metrics import compute_stats
from evaluation.kde import plot_kde


# ساخت سیستم
stm = STM(window_size=5)
ltm = LTM()
hybrid = HybridRetriever(stm, ltm)

# اجرای ارزیابی
scores = evaluate_retriever(
    retriever_function=hybrid.retrieve_and_generate,
    name="LTM Retrieve",
    max_samples=30
)

# محاسبه آمار
stats = compute_stats(scores)
print("\nFinal Stats:", stats)

# رسم KDE
plot_kde(scores, "LTM Retrieve KDE")