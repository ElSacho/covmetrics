# Covmetrics: conditional coverage metrics

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
[Conditional Coverage Diagnostics for Conformal Prediction](https://arxiv.org/abs/2512.11779).
Please cite us if you use this repository for research purposes.

## Installation

Work in progress: for now you can use the ERT file to use our metrics, it will be pip installable soon! 
<!-- Covmetrics is available via
```bash
pip install covmetrics
``` -->

## Using conditional coverage metrics

For a quick usage, you can evaluate a metric as follows:
```python
from covmetrics import ERT 

ERT_value = ERT().evaluate(x, cover, alpha, n_splits = 5)

```

Where the object "x" is a feature vector (numpy, torch or dataframe), and cover is a vector with 0's or 1's

The default classifier used to classify the outputs is a LightGBM classifier.
You can change this by replacing the model class of the classifier:

```python
from covmetrics import ERT 
from sklearn.linear_model import LogisticRegression

ERT_estimator = ERT(model_cls=LogisticRegression)
```

We recommend using our k-folds pre-implemented version to evaluate the conditional miscoverage by doing :


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

## Other metrics

The WSC metric is a vectorized version of the original github : Original code from https://github.com/Shai128/oqr.

```python
from covmetrics import WSC 

WSC_estimator = WSC().evaluate(x_test, cover_test)
```
For the CovGap metric, or the WeightedCovGap one, it can be estimated as: 

```python
from covmetrics import CovGap 

CovGap_estimator = CovGap().evaluate(x_test, cover_test, alpha=alpha, weighted=True)
```

Similar import can be used to use the metrics SSC, FSC, EOC, HSIC and PearsonCorrelation.

The HSIC metric has been built upon the original code from: https://github.com/danielgreenfeld3/XIC.


## Contributors
- Sacha Braun
- David Holzmüller