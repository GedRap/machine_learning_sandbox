# Uses elastic net model and grid approach to find the most optimal values
from __future__ import division

from sklearn.datasets import load_boston
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import ElasticNet



alpha_values = []
for a in range(1, 101):
    alpha_values.append(a / 10)

parameters = { 'alpha': alpha_values }

if __name__ == "__main__":
    grid_search = GridSearchCV(ElasticNet(), parameters, n_jobs=-1, verbose=1, cv=5)
    data = load_boston()
    x = data.data
    y = data.target
    grid_search.fit(x, y)

    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    # Best score: 0.484923639314 with alpha 0.5
    print "Best score: {bs} with alpha {alpha}".format(bs=best_score, alpha=best_params['alpha'])
