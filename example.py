import numpy as np
import utils as ut
import config as cfg
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


X, y, ids = ut.load_ndvi_uts(cfg.data_path, ut.as_list(2015), cfg.balance_flag)


mdl = LogisticRegression(random_state=cfg.random_state)
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
scaler = preprocessing.StandardScaler()

clf = Pipeline([("imputer", imputer), ("scalar", scaler), ("mdl", mdl)])
auc_cv = cross_val_score(clf, X, y, cv=cfg.cv, scoring=cfg.scoring)
print(np.mean(auc_cv))
