
print(__doc__)

import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time

n_components = 2
(fig, subplots) = plt.subplots(2, 2, figsize=(15, 8))

#data = pd.read_csv('crop_tsc_balanced_imputed_2015.csv', index_col=None, header=None)
#X = data.iloc[:,0:8]
#y = data.iloc[:,9]

data = pd.read_csv('crop_tsc_balanced_imputed_rbf_2015.csv', index_col=None, header=None)
X = data.iloc[:,0:100]
y = data.iloc[:,100]


red = y == 0
green = y == 1

print(green)


ax = subplots[0][0]

t0 = time()
tsne = manifold.Isomap(n_components=n_components)
Y = tsne.fit_transform(X)
t1 = time()
ax.scatter(Y[red, 0], Y[red, 1], c="r")
ax.scatter(Y[green, 0], Y[green, 1], c="g")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis('tight')

plt.savefig('isomap-rbf.png')