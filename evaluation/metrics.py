import numpy as np

def compute_stats(scores):

    mean = np.mean(scores)
    median = np.median(scores)
    std = np.std(scores)
    var = np.var(scores)

    return {
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "var": round(var, 2)
    }