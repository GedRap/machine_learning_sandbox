from __future__ import division

from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn. cross_validation import cross_val_score
import pandas as pd
import numpy as np

from datetime import datetime

data = pd.read_csv('data/communities.data', header=None, na_filter=True, na_values='?')

# data not used for predictions such as county name
non_predictive_data = data[list(data.columns)[0:4]]
# features used for predictions
features = data[list(data.columns)[5:126]]
# value to be predicted (number of violent crimes)
goal = data[list(data.columns)[127]]

# plenty of values are missing in the end of features vector (at indices around 115)
# therefore we will eliminate columns where at least one sample has missing data
features = features.dropna(axis=1)

alpha_values = []
for a in range(1, 10001):
    alpha_values.append(a / 100)

print "Started at " + str(datetime.now())

estimator_ridge = RidgeCV(alphas=alpha_values, cv=3)
estimator_ridge.fit(features, goal)
scores = cross_val_score(Ridge(alpha=estimator_ridge.alpha_), features, goal, cv=5)
print "Ridge alpha " + str(estimator_ridge.alpha_)
print str(np.mean(scores))
print scores

estimator_lasso = LassoCV(alphas=alpha_values, cv=3)
estimator_lasso.fit(features, goal)
scores = cross_val_score(Lasso(alpha=estimator_lasso.alpha_), features, goal, cv=5)
print "Lasso alpha " + str(estimator_lasso.alpha_)
print str(np.mean(scores))
print scores


estimator_elastic_net = ElasticNetCV(alphas=alpha_values, cv=3, n_jobs=-1)
estimator_elastic_net.fit(features, goal)
scores = cross_val_score(ElasticNet(alpha=estimator_elastic_net.alpha_), features, goal, cv=5)
print "ElasticNet alpha " + str(estimator_elastic_net.alpha_)
print str(np.mean(scores))
print scores

print "Finished at " + str(datetime.now())