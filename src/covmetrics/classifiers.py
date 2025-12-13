import numpy as np
from lightgbm import LGBMClassifier
import sklearn
from probmetrics.calibrators import get_calibrator
import multiprocessing as mp

class CheapLGBMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def _fit_model(self, idxs):
        m = LGBMClassifier(n_estimators=1_000, learning_rate=0.04, subsample=0.75, subsample_freq=1, num_leaves=50,
                           random_state=0, early_stopping_round=100, min_child_samples=40, min_child_weight=1e-7,
                           n_jobs=1, verbosity=0)
        return m.fit(self.X_.take(idxs[0], 0), self.y_[idxs[0]], eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]]))

    def fit(self, X, y):
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)],
                                   axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))
