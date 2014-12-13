from __future__ import division

from sklearn.linear_model import Ridge
from sklearn. cross_validation import cross_val_score
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split
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

features_train, features_test, goal_train, goal_test = train_test_split(features, goal)

estimator = Ridge(alpha=2.6)
estimator.fit(features_train, goal_train)

print "Score on training data: " + str(estimator.score(features_train, goal_train))

predicted = estimator.predict(features_test)

print "Explained variance score"
print explained_variance_score(predicted, goal_test)

print "Mean absolute error"
print mean_absolute_error(predicted, goal_test)

print "Mean squared error"
print mean_squared_error(predicted, goal_test)

print "R^2 score"
print r2_score(predicted, goal_test)

# Score on training data: 0.68621807808
# Explained variance score
# 0.557129585719
# Mean absolute error
# 0.0910155466665
# Mean squared error
# 0.0175901455272
# R^2 score
# 0.555938187187