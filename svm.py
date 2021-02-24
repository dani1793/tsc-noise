print(__doc__)


# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

data = pd.read_csv('crop_tsc_balanced_imputed_2015.csv', index_col=None, header=None)
X = data.iloc[:,0:8]
Y = data.iloc[:,9]



# figure number
fignum = 1

# fit the model
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
        
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    print("svm created for kernel type %s"%(kernel))
    print(clf.support_vectors_.shape)
    #plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
    #            edgecolors='k')

# plt.savefig('SVM-comparison.png')
print("done")
