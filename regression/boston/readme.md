# Boston real estate pricing predictor

Predicts median price for a house in town.
Uses various regression methods based on features such as crime rate, schools, access to highway.

## Predictors used

* Standard linear regression
* Lasso
* Ridge
* Elastic net

For Lasso, Ridge and Elastic net predictors, I used multiple different penalization (alpha) values. Elastic net was performing
the best with alpha=1, mean score 0.47 meaning it explains 47% of the variance. I performed 5-folded cross validation to
calculate the score.

## Output

Scores for predictors tested (mean and score for every cross test).

```
/usr/bin/python2.7 boston_real_estate_pricing.py

Standard linear regression
0.350741350933
[ 0.63861069  0.71334432  0.58645134  0.07842495 -0.26312455]

Lasso regression
alpha=0.1
0.401031199074
[ 0.67465461  0.73260743  0.63221137  0.09407457 -0.12839199]
alpha=0.2
0.409002048878
[ 0.6667744   0.7297584   0.59222736  0.13530489 -0.0790548 ]
alpha=0.3
0.418925062486
[ 0.65948469  0.7201878   0.55571611  0.17036754 -0.01113082]
alpha=0.4
0.425854992461
[ 0.65034118  0.70886542  0.51566133  0.20320739  0.05119965]

Ridge regression
alpha=1
0.386979054732
[ 0.66040085  0.7408794   0.62881062  0.0843547  -0.1795503 ]
alpha=2
0.397113889907
[ 0.66652568  0.74024496  0.64112711  0.09102672 -0.15335502]
alpha=3
0.40281429157
[ 0.66892491  0.73874286  0.64637348  0.09827069 -0.13824049]
alpha=4
0.407006244511
[ 0.66993288  0.73749458  0.6488719   0.10571389 -0.12698202]

Elastic net regularization
alpha=1
0.472705248975
[ 0.57006633  0.66282788  0.40313917  0.45896157  0.26853131]
alpha=2
0.432414630219
[ 0.49541265  0.6353668   0.28232168  0.45678415  0.29218787]
alpha=3
0.392994145044
[ 0.42615297  0.61712524  0.19468152  0.43458826  0.29242273]
alpha=4
0.364213346141
[ 0.37215382  0.60042231  0.1362107   0.41063795  0.30164195]

```


## Data set

Used Boston data set included with scikit-learn which includes 506 13-dimensional samples.

### Data set description

Boston House Prices dataset

Notes
------
Data Set Characteristics:

:Number of Instances: 506
:Number of Attributes: 13 numeric/categorical predictive
:Median Value (attribute 14) is usually the target

:Attribute Information (in order)
- CRIM     per capita crime rate by town
- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS    proportion of non-retail business acres per town
- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX      nitric oxides concentration (parts per 10 million)
- RM       average number of rooms per dwelling
- AGE      proportion of owner-occupied units built prior to 1940
- DIS      weighted distances to five Boston employment centres
- RAD      index of accessibility to radial highways
- TAX      full-value property-tax rate per $10,000
- PTRATIO  pupil-teacher ratio by town
- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT    % lower status of the population
- MEDV     Median value of owner-occupied homes in $1000's

:Missing Attribute Values: None
:Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
http://archive.ics.uci.edu/ml/datasets/Housing

This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\nprices and the demand for clean air', J. Environ. Economics & Management,\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\npages 244-261 of the latter.\n\nThe Boston house-price data has been used in many machine learning papers that address regression\nproblems.   \n     \n**References**\n\n   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n   - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)