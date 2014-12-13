from __future__ import division

from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn. cross_validation import cross_val_score
from sklearn.decomposition import PCA
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

features = features.dropna(axis=1)

alpha_values = []
for a in range(1, 1001):
    alpha_values.append(a / 10)

started_at = str(datetime.now())

best_score = -1
best_scores = []
best_alpha = 0
best_estimator = None
best_pca_n = 0

for n_features in [10, 25, 50, 75, 99]:
    print "PCA n_components=" + str(n_features)

    pca = PCA(n_components=n_features)
    reduced_features = pca.fit_transform(features)

    estimator_ridge = RidgeCV(alphas=alpha_values, cv=3)
    estimator_ridge.fit(reduced_features, goal)
    scores = cross_val_score(Ridge(alpha=estimator_ridge.alpha_), features, goal, cv=5)
    print "Ridge alpha " + str(estimator_ridge.alpha_)
    print str(np.mean(scores))
    print scores

    if np.mean(scores) > best_score:
        best_score = np.mean(scores)
        best_scores = scores
        best_alpha = estimator_ridge.alpha_
        best_estimator = "Ridge"
        best_pca_n = n_features

    estimator_lasso = LassoCV(alphas=alpha_values, cv=3)
    estimator_lasso.fit(features, goal)
    scores = cross_val_score(Lasso(alpha=estimator_lasso.alpha_), features, goal, cv=5)
    print "Lasso alpha " + str(estimator_lasso.alpha_)
    print str(np.mean(scores))
    print scores

    if np.mean(scores) > best_score:
        best_score = np.mean(scores)
        best_scores = scores
        best_alpha = estimator_lasso.alpha_
        best_estimator = "Lasso"
        best_pca_n = n_features


    estimator_elastic_net = ElasticNetCV(alphas=alpha_values, cv=3, n_jobs=-1)
    estimator_elastic_net.fit(features, goal)
    scores = cross_val_score(ElasticNet(alpha=estimator_elastic_net.alpha_), features, goal, cv=5)
    print "ElasticNet alpha " + str(estimator_elastic_net.alpha_)
    print str(np.mean(scores))
    print scores

    if np.mean(scores) > best_score:
        best_score = np.mean(scores)
        best_scores = scores
        best_alpha = estimator_elastic_net.alpha_
        best_estimator = "EN"
        best_pca_n = n_features

print "Started at " + started_at
print "Finished at " + str(datetime.now())

print "Best score " + str(best_score)
print best_scores
print "Best alpha " + str(best_alpha)
print str(best_estimator)

print "Number of features: " + str(best_pca_n)

# Best score 0.655822655819
# [ 0.61322066  0.64227616  0.71648529  0.67040043  0.63673074]
# Best alpha 2.6
# Ridge
# Number of features: 99