
print(__doc__)

import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time

n_samples = 10000
n_components = 2
(fig, subplots) = plt.subplots(2, 6, figsize=(15, 8))
perplexities = [5, 10, 20, 30, 50, 200, 250, 500, 1000]

#data = pd.read_csv('crop_tsc_balanced_imputed_2015.csv', index_col=None, header=None)
#X = data.iloc[:,0:8]
#y = data.iloc[:,9]

data = pd.read_csv('crop_tsc_balanced_imputed_rbf_2015.csv', index_col=None, header=None)
X = data.iloc[:,0:100]
y = data.iloc[:,100]


red = y == 0
green = y == 1

print(green)

for i, perplexity in enumerate(perplexities):
    ax = subplots[0 if i < 6 else 1][i % 6]

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, learning_rate = 50, n_iter= 500, init='random',
                         random_state=0, perplexity=perplexity, n_jobs=4)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(Y[red, 0], Y[red, 1], c="r")
    ax.scatter(Y[green, 0], Y[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')

plt.savefig('TSNE-rbf-sample-Learning-rate-50-iterations-500.png')