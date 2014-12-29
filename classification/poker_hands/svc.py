from __future__ import division

import sys

import pandas as pd
import numpy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':

    data = pd.read_csv('data/poker-hand-training-15000.data', header=None)
        
    features = data[list(data.columns)[0:10]]
    target = data[list(data.columns)[10]]

    enc = OneHotEncoder()
    suits_by_row = []
    for row in range(0, len(features[0].values)):
        suits = [features[0].values[row], features[2].values[row], features[4].values[row], features[6].values[row], features[8].values[row]]
        suits_by_row.append(suits)

    enc.fit(suits_by_row)

    suits_enc = []
    for row in suits_by_row:
        suits_row_enc = enc.transform([row]).toarray()
        suits_enc.append(suits_row_enc)


    rank_by_row = []
    for row in range(0, len(features[0].values)):
        ranks = [features[1].values[row], features[3].values[row], features[5].values[row], features[7].values[row], features[9].values[row]]
        rank_by_row.append(ranks)

    enc.fit(rank_by_row)

    ranks_enc = []
    for row in rank_by_row:
        rank_row_enc = enc.transform([row]).toarray()
        ranks_enc.append(rank_row_enc)

    features_enc = []
    for row in range(0, len(ranks_enc)):
        features_enc_row = numpy.concatenate([ranks_enc[row][0], suits_enc[row][0]])
        features_enc.append(features_enc_row)

    
    features_train, features_test, target_train, target_test = train_test_split(features_enc, target)
    
    pipeline = Pipeline([('clf', SVC(kernel='rbf', gamma=0.01, C=100, cache_size=1300))])

    parameters = {
        'clf__gamma': (0.1, 1, 10),
        'clf__C': (1, 10, 100)
    }
    
    grid_search = GridSearchCV(pipeline, parameters, verbose=1, n_jobs=-1, scoring='precision')
    grid_search.fit(features_train, target_train)
    
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
        
    print "Test"
    print "*" * 20
    predictions = grid_search.predict(features_test)
    print classification_report(target_test, predictions)
    print confusion_matrix(target_test, predictions)


# Best score: 0.734
# Best parameters set:
# 	clf__C: 10
# 	clf__gamma: 0.1
# Test
# ********************
#              precision    recall  f1-score   support
#
#           0       0.92      0.96      0.94      1863
#           1       0.82      0.91      0.86      1586
#           2       0.42      0.08      0.14       189
#           3       0.50      0.02      0.05        81
#           4       0.00      0.00      0.00        16
#           5       1.00      0.29      0.44         7
#           6       0.00      0.00      0.00         6
#           8       0.00      0.00      0.00         1
#           9       0.00      0.00      0.00         1
#
# avg / total       0.84      0.86      0.84      3750
#
# [[1782   80    0    0    1    0    0    0    0]
#  [ 134 1440   11    1    0    0    0    0    0]
#  [   1  171   16    1    0    0    0    0    0]
#  [   0   70    9    2    0    0    0    0    0]
#  [  15    1    0    0    0    0    0    0    0]
#  [   5    0    0    0    0    2    0    0    0]
#  [   0    4    2    0    0    0    0    0    0]
#  [   1    0    0    0    0    0    0    0    0]
#  [   1    0    0    0    0    0    0    0    0]]





