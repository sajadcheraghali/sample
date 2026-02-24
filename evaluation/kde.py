import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def plot_kde(scores, title):

    scores = np.array(scores).reshape(-1,1)

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde.fit(scores)

    x = np.linspace(0,10,100).reshape(-1,1)
    log_dens = kde.score_samples(x)

    plt.plot(x, np.exp(log_dens))
    plt.title(title)
    plt.xlabel("Relevance Score")
    plt.ylabel("Density")
    plt.show()