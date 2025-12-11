from sklearn.model_selection import KFold
import torch
import numpy as np
import pandas as pd
import warnings

def check_n_splits(n_splits):
    """
    Checks if n_splits is a valid integer greater than 0.
    
    Raises:
        TypeError: If n_splits is not an integer.
        ValueError: If n_splits is less than 1.
    """
    if not isinstance(n_splits, int):
        raise TypeError(f"n_splits must be an integer, got {type(n_splits).__name__}")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1 for cross-validation.")
    return True


def clip_max(x, val_max):
    if isinstance(x, torch.Tensor):
        clipped = torch.clamp(x, min=None, max=val_max)
        return clipped

    elif isinstance(x, np.ndarray):
        clipped = np.clip(x, x, val_max)
        return clipped
    
def clip_min(x, val_min):
    if isinstance(x, torch.Tensor):
        clipped = torch.clamp(x, min=val_min, max=None)
        return clipped

    elif isinstance(x, np.ndarray):
        clipped = np.clip(x, val_min, x)
        return clipped

def brier_score(pred_proba, cover):
    return (pred_proba - cover)**2

def logloss(pred_proba, cover):
    eps=1e-6
    if isinstance(cover, torch.Tensor):
        if not isinstance(pred_proba, torch.Tensor):
            pred_proba = torch.tensor(pred_proba, dtype=torch.float32)
        pred_proba = torch.clip(pred_proba, eps, 1-eps)
        return cover * torch.log(pred_proba) + (1-cover)*torch.log(1-pred_proba)
    else:
        pred_proba = np.clip(pred_proba, eps, 1-eps)
        return - cover * np.log(pred_proba) - (1-cover)*np.log(1-pred_proba)

def make_L1_miscoverage(alpha):
    def L1_miscoverage(pred_proba, cover):
        threshold = 1 - alpha
        
        out = cover * 0
        
        pos = pred_proba < threshold
        neg = pred_proba > threshold
    
        out[pos] = (threshold - cover)[pos]
        out[neg] = -(threshold - cover)[neg]

        return - out
    
    return L1_miscoverage
           
class ERT:
    def __init__(self, model_cls, **model_kwargs):
        """
        model_cls: the class of the model (e.g., RandomForestClassifier, CatBoostClassifier)
        model_kwargs: keyword arguments to initialize the model
        """
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.model = self.init_model()
        self.fitted = False
        self.added_losses = None

    def init_model(self):
        """Re-initialize the model."""
        self.model = self.model_cls(**self.model_kwargs)
        return self.model
    
    def fit(self, x_train, cover_train, x_stop=None, cover_stop=None, **fit_kwargs):
        # check_tabular(x_train)
        # check_cover(cover_train)
        # check_consistency(cover_train, x_train)
        if x_stop is not None:
            # check_tabular(x_stop)
            # check_cover(cover_stop)
            # check_consistency(cover_stop, x_stop)
            pass

        if cover_stop is not None:
            self.model.fit(x_train, cover_train, X_val=x_stop, y_val=cover_stop, **fit_kwargs)
        else:
            self.model.fit(x_train, cover_train, **fit_kwargs)
       
        self.fitted = True
    
    def get_conditional_prediction(self, x):
        eps = 1e-5

        if hasattr(self.model, "predict_proba"):
            output = self.model.predict_proba(x)[:, 1]
        else:
            warnings.warn("The model does not support predict_proba. Using predict instead.")
            output = self.model.predict(x)
            output = np.clip(output, eps, 1 - eps)

        if isinstance(x, pd.DataFrame):
            output = pd.Series(output, index=x.index)

        if isinstance(x, torch.Tensor):
            output = torch.tensor(output, dtype=x.dtype)

        return output


    def get_binary_prediction(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        output = self.model.predict(x)
        if isinstance(x, pd.DataFrame):
            output = pd.Series(output, index=x.index)
        return output
        
    def add_loss(self, loss):
        if self.added_losses is None:
            self.added_losses = [loss]
        else:
            self.added_losses.append(loss)

    def make_losses(self, alpha):
        self.tab_losses = [brier_score, make_L1_miscoverage(alpha), logloss]
        if self.added_losses is not None:
            for new_loss in self.added_losses:
                self.tab_losses.append(new_loss)
       
    def evaluate_all(self, x, cover, alpha, n_splits = None, val_splits=None, random_state=42, underconfidence=False, overconfidence=False, **fit_kwargs):
        # check_tabular(x)
        # check_cover(cover)
        # check_consistency(cover, x)
        # check_alpha(alpha)

        self.make_losses(alpha)

        ERN_values = {"ERN_"+loss.__name__: [] for loss in self.tab_losses}
        if underconfidence:
            for loss in self.tab_losses:
                ERN_values["ERN_underconfident_"+loss.__name__] = []
        if overconfidence:
            for loss in self.tab_losses:
                ERN_values["ERN_overconfident_"+loss.__name__] = []
        
        if n_splits is not None:
            check_n_splits(n_splits)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x

            for train_index, test_index in kf.split(x_np):
                if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                else:
                    x_train, x_test = x[train_index], x[test_index]
                if isinstance(cover, pd.DataFrame) or isinstance(cover, pd.Series):
                    cover_train, cover_test = cover.iloc[train_index], cover.iloc[test_index]
                else:
                    cover_train, cover_test = cover[train_index], cover[test_index]

                self.init_model()
                if val_splits is not None:
                    n_val = int(len(x_train)*val_splits)
                    x_val = x_train[:n_val]
                    cover_val = cover_train[:n_val]
                    x_train = x_train[n_val:]
                    cover_train = cover_train[n_val:]
                    self.model.fit(x_train, cover_train, X_val=x_val, y_val=cover_val, **fit_kwargs)
                else:
                    self.model.fit(x_train, cover_train, **fit_kwargs)

                pred_test = self.get_conditional_prediction(x_test)

                for loss in self.tab_losses:
                    ERN_loss = evaluate_with_predictions(pred_test, cover_test, alpha, loss=loss)
                    ERN_values["ERN_"+loss.__name__].append(ERN_loss)

                if underconfidence:
                    underconfident_pred_test = clip_min(pred_test, 1-alpha)
                    for loss in self.tab_losses:
                        ERN_loss = evaluate_with_predictions(underconfident_pred_test, cover_test, alpha, loss=loss)
                        ERN_values["ERN_underconfident_"+loss.__name__].append(ERN_loss)

                if overconfidence:
                    overconfident_pred_test = clip_max(pred_test, 1-alpha)
                    for loss in self.tab_losses:
                        ERN_loss = evaluate_with_predictions(overconfident_pred_test, cover_test, alpha, loss=loss)
                        ERN_values["ERN_overconfident_"+loss.__name__].append(ERN_loss)
                    
            results = {key: np.mean(values) for key, values in ERN_values.items()}

            return results
        else:
            if not self.fitted:
                raise Exception("You need to first fit the model. You can evaluate with cross validation using cross_val == True")
        
        pred = self.get_conditional_prediction(x)
        
        for loss in self.tab_losses:
            ERN_loss = evaluate_with_predictions(pred, cover, alpha, loss=loss)
            results["ERN_"+loss.__name__].append(ERN_loss)

        if underconfidence:
            underconfident_pred_test = clip_min(pred, 1-alpha)
            for loss in self.tab_losses:
                ERN_loss = evaluate_with_predictions(underconfident_pred_test, cover, alpha, loss=loss)
                ERN_values["ERN_underconfident_"+loss.__name__].append(ERN_loss)

        if overconfidence:
            overconfident_pred_test = clip_max(pred, 1-alpha)
            for loss in self.tab_losses:
                ERN_loss = evaluate_with_predictions(overconfident_pred_test, cover, alpha, loss=loss)
                ERN_values["ERN_overconfident_"+loss.__name__].append(ERN_loss)

        return results
   
  
    def evaluate(self, x, cover, alpha, n_splits = None, random_state=42, loss=brier_score, **fit_kwargs):
        # check_tabular(x)
        # check_cover(cover)
        # check_consistency(cover, x)
        # check_alpha(alpha)
        
        if n_splits is not None:
            check_n_splits(n_splits)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            ERN_values = []
                        
            x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x

            for train_index, test_index in kf.split(x_np):
                x_train, x_test = x[train_index], x[test_index]
                cover_train, cover_test = cover[train_index], cover[test_index]

                self.init_model()
                self.model.fit(x_train, cover_train, **fit_kwargs)

                pred_test = self.get_conditional_prediction(x_test)

                ERN_loss = evaluate_with_predictions(pred_test, cover_test, alpha, loss=loss)
                ERN_values.append(ERN_loss)

            results = {"ERN_"+loss.__name__: np.mean(ERN_values)}
            return results
            
        else:
            if not self.fitted:
                raise Exception("You need to first fit the model. You can evaluate with cross validation using cross_val == True")
        
        pred = self.get_conditional_prediction(x)
        
        ERN_loss = evaluate_with_predictions(pred, cover, alpha, loss=loss)
        results = {"ERN_"+loss.__name__: ERN_loss}

        return results

def evaluate_with_predictions(pred, cover, alpha, loss=brier_score):
    loss_pred = loss(pred, cover)
    loss_bayes = loss(1-alpha, cover)
    return np.mean(np.array(loss_bayes)) - np.mean(np.array(loss_pred))