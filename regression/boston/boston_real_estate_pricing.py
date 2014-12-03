# Test several different models with hardcoded parameter values

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn. cross_validation import cross_val_score
import matplotlib.pyplot as plt

data = load_boston()
x = data.data
y = data.target

# standard linear regression
print "Standard linear regression"

linear = LinearRegression()
# [ 0.63861069  0.71334432  0.58645134  0.07842495 -0.26312455]
linear_scores = cross_val_score(linear, x, y, cv=5)
print linear_scores.mean()
print linear_scores

# for many parameters, beta will be zero
# this way, the algorithm eliminates non-relevant and least-relevant features
print "Lasso regression"

for alpha in range(1,5):
    alpha = float(alpha) / 10
    lasso = Lasso(alpha)
    lasso_scores = cross_val_score(lasso, x, y, cv=5)
    print "alpha={a}".format(a=alpha)
    print lasso_scores.mean()
    print lasso_scores


# different than Lasso, penalizes but still beta for most features will remain > 0
print "Ridge regression"

for alpha in range(1,5):
    ridge = Ridge(alpha)
    ridge_scores =cross_val_score(ridge, x, y, cv = 5)
    print "alpha={a}".format(a=alpha)
    print ridge_scores.mean()
    print ridge_scores

# combination of ridge and Lasso
print "Elastic net regularization"

for alpha in range(1,5):
    elastic_net = ElasticNet(alpha)
    elastic_net_scores =cross_val_score(elastic_net, x, y, cv = 5)
    print "alpha={a}".format(a=alpha)
    print elastic_net_scores.mean()
    print elastic_net_scores

# best performing regressor for this data set was Elastic net with alpha=1
# with score = 0.472705248975
# draw scatter plot for values predicted with this regressor

print "Showing scatter plot for elastic net with alpha = 1"

elastic_net = ElasticNet(1)
elastic_net.fit(x, y)
predicted_y = elastic_net.predict(x)

fig = plt.figure()
plt.scatter(y, predicted_y, alpha=0.3, )
fig.suptitle('Boston real estate pricing', fontsize=20)
plt.figtext(.5,.9,'Elastic net regularization, alpha=1', fontsize=15, ha='center')
plt.xlabel('Actual value, $1000s', fontsize=18)
plt.ylabel('Predicted value, $1000s', fontsize=18)
plt.show()
