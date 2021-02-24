from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
import pandas as pd

data = pd.read_csv('crop_tsc_balanced_imputed_2015.csv', index_col=None, header=None)
X = data.iloc[:,0:8]
y = data.iloc[:,9]

rbf_feature = RBFSampler(gamma=10, random_state=1)
X_features = rbf_feature.fit_transform(X)
clf = SGDClassifier(max_iter=1000)
clf.fit(X_features, y)
print(clf.score(X_features, y))

df = pd.DataFrame(X_features)
df.insert(100,100, y)
print(df.head())
df.to_csv("crop_tsc_balanced_imputed_rbf_2015.csv", header=False, index=False)
