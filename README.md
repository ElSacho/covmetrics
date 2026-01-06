# Covmetrics: conditional coverage metrics

This package (PyTorch-based) currently contains different conditional coverage metrics, including our metric ERT (Excess risk of the target coverage).

It accompanies our papers
[Conditional Coverage Diagnostics for Conformal Prediction](https://arxiv.org/abs/2512.11779).
Please cite us if you use this repository for research purposes.

## Installation

Covmetrics is available via
```bash
pip install covmetrics
```

## Using conditional coverage metrics

For a quick usage, you can evaluate a metric as follows:
```python
from covmetrics import ERT 

ERT_value = ERT().evaluate(x, cover, alpha)

```

Where the object "x" is a feature vector of shape (n_samples, n_features) (numpy, torch or dataframe), and cover is a vector of shape (n_samples,) with 0's or 1's

The default classifier used to classify the outputs is a LightGBM classifier.
You can change this by replacing the model class of the classifier:

```python
from covmetrics import ERT 
from sklearn.linear_model import LogisticRegression

ERT_estimator = ERT(model_cls=LogisticRegression)
```

We recommend using our k-folds pre-implemented version to evaluate the conditional miscoverage by doing (default value is 5):


```python
ERT_value = ERT_estimator.evaluate(x_test, cover_test, alpha, n_splits = 5)
```

But you can choose between training the classifier with some data and using it on other doing the following:

```python
ERT_estimator.fit(x_train, cover_train)
ERT_value = ERT_estimator.evaluate(x_test, cover_test, alpha, n_splits=0)
```

## Modifying the loss function

The default loss used to evaluate the classifier provides a lower bound on the $L_1$-ERT. You can change the loss by doing :

```python
ERT_estimator.evaluate(x_test, cover_test, alpha, loss=your_loss)
```

The package already provides several losses functions to evaluate your models. You can import them as follows:

```python
from covmetrics.losses import (
    brier_score,
    logloss,
    L1_miscoverage,
    brier_score_over,
    L1_miscoverage_over,
    logloss_over,
    brier_score_under,
    logloss_under,
    L1_miscoverage_under
)
```

If you want to evaluate more losses at the same time, you can use 
```python
ERT_value = ERT_estimator.evaluate_multiple_losses(x_test, cover_test, alpha, all_losses = List_of_all_your_losses)
```
Which returns a dictionnary with all evaluated losses .
By default, if all_losses=None, the metrics evaluated are the $L_1$-ERT, $L_2$-ERT and KL-ERT.

## Using a custom classifier

You can also use your own classifier with ERT. To do this, define a class with the following methods:

- `__init__(self, **model_kwargs)`: Initialize your model with any required parameters.  
- `fit(self, X, y, **fit_kwargs)`: Train the model on your data.  
- `predict_proba(self, X)`: Return the predicted probabilities for each class.  

Once your class is defined, you can instantiate and evaluate it with ERT as follows:

```python
ERT_estimator = ERT(your_model_class, one_argument=k, another_argument="p")
ERT_value = ERT_estimator.evaluate(x_test, cover_test, alpha, one_fit_argument=m, another_fit_argument="M")
```

## Evaluating Conditional Coverage Rules

You can evaluate conditional coverage rules by providing `alpha` as an array that matches the type and length of `cover`. For example, if `cover` is a PyTorch tensor, `alpha` should also be a tensor; if `cover` is a NumPy array, `alpha` should be a NumPy array.  

Each value of `alpha` must be between 0 and 1 and represents the **conditional miscoverage level**, meaning:

$$
\mathbb{P}(Y \in C(X) \mid X) = 1 - \alpha(X)
$$

Example usage:

```python
ERT_estimator = ERT()
tab_alpha = torch.ones(len(cover_test)) * 0.9 # if cover_test is a torch.Tensor
ERT_value = ERT_estimator.evaluate(x_test, cover_test, alpha=tab_alpha)
```


## Other metrics

Other metrics implemented metrics are: 

- WSC (Worst slab coverage).
- FSC (Feature-stratified coverage).
- CovGap.
- WeightedCovGap.
- SSC (Size-stratified coverage).
- EOC (Equal opportunity of coverage).
- Pearson's Correlation.
- HSIC's Correlation.

The WSC metric is a vectorized version of the original github : Original code from https://github.com/Shai128/oqr.

```python
from covmetrics import WSC 

WSC_value = WSC().evaluate(x_test, cover_test)
```
For the CovGap metric, or the WeightedCovGap one, it can be estimated as: 

```python
from covmetrics import CovGap 

CovGap_value = CovGap().evaluate(x_test, cover_test, alpha=alpha, weighted=True)
```

Similar import can be used to use the metrics SSC, FSC, EOC, HSIC and PearsonCorrelation.

The HSIC metric has been built upon the original code from: https://github.com/danielgreenfeld3/XIC.


## Contributors
- Sacha Braun
- David Holzmüller