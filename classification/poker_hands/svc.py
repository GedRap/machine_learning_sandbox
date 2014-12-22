from __future__ import division

import sys

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':

    data = pd.read_csv('data/poker-hand-training-true.data', header=None)
        
    features = data[list(data.columns)[0:10]]
    target = data[list(data.columns)[10]]
    
    features_train, features_test, target_train, target_test = train_test_split(features, target)
    
    pipeline = Pipeline([('clf', SVC(kernel='rbf', gamma=0.01, C=100))])
    # parameters = {
    #     'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1, 1.5, 2, 2.5, 3),
    #     'clf__C': (0.1, 0.3, 1, 3, 10, 30, 50, 75, 100)
    # }
    
    parameters = {
        'clf__gamma': (0.01, 0.1, 0.5, 1, 2, 5, 10),
        'clf__C': (0.1, 1, 3, 10, 50, 100)
    }
    
    grid_search = GridSearchCV(pipeline, parameters, verbose=1, scoring='precision')
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



