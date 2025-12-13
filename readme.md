# Covmetrics: conditional coverate metrics

This package (PyTorch-based) currently contains different conditional coverage metrics:
- ERT (Excess risk of the target coverage).
- WSC (Worst slab coverage).
- FSC (Feature-stratified coverage).
- CovGap.
- WeightedCovGap.
- SSC (Size-stratified coverage).
- EOC (Equal opportunity of coverage).
- Pearson's Correlation.
- HSIC's Correlation.

It accompanies our papers
[Conditional Coverage Diagnostics for Conformal Prediction]().
Please cite us if you use this repository for research purposes.

## Installation

Covmetrics is available via
```bash
pip install covmetrics
```

## Using conditional coverage metrics

For a quick usage, you can evaluate a metric as follows:
```python
from covmetrics.ERT import ERT 

ERT_value = ERT().evaluate(x, cover, alpha, n_splits = 5)

```
Where the object "x" is a feature vector (numpy, torch or dataframe), and cover is a vector with 0's or 1's

The default classifier used to classiify the outputs is a LightGBM Classifier.
You can change this by replacing the model class of the classifier 

```python
from covmetrics.ERT import ERT 
from sklearn.linear_model import LogisticRegression

ERT_estimator = ERT(model_cls=LogisticRegression)

```
We recommand using our k-folds pre-implement version to evaluate the conditional miscoverage by doing :


```python
ERT_estimator.evaluate(x_test, cover_test, alpha, n_splits = 5)
```

But you can choose between training the classifier with some data and using it on other doing the following:

```python
ERT_estimator.fit(x_train, cover_train)
ERT_estimator.evaluate(x_test, cover_test, alpha)
```

The default loss used to evaluate the classifier provides a lower bound on the $L_1$-ERT. You can change the loss by doing :

```python
ERT_estimator.evaluate(x_test, cover_test, alpha, loss=your_loss)
```

If you want to evaluate more losses at the same time, you can use 
```python
ERT_estimator.evaluate_multiple_losses(x_test, cover_test, alpha, all_losses_to_evaluate = List_of_all_your_losses)
```
Which returns a dictionnary with all evaluated losses .
By default, if all_losses_to_evaluate=None, the metrics evaluated are the $L_1$-ERT, $L_2$-ERT and KL-ERT.


## Contributors
- Sacha Braun
- David Holzmüller