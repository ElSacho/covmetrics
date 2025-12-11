import torch 
from typing import Literal, Optional
import pandas as pd
from tqdm import tqdm
from scipy import stats
import numpy as np
from math import ceil
from typing import Iterable, Union, List, Tuple, Dict
import math
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  


def check_alpha(alpha):
    """Ensure alpha is a float in (0,1)."""
    try:
        alpha = float(alpha)
    except Exception:
        raise TypeError("alpha must be a float or int.")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1 (inclusive).")
    
def check_delta(alpha):
    """Ensure delta is a float in (0,1)."""
    try:
        alpha = float(alpha)
    except Exception:
        raise TypeError("delta must be a float or int.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1 (delta).")

def check_tabular_1D(x):
    """
    Check that x is a valid 1D tabular array/vector.
    - Must be 1D
    - No NaN values
    - No ±Inf values
    - All entries are finite numbers
    """
    # Check dimensionality
    if x.ndim != 1:
        raise ValueError(f"x must be 1D (tabular vector), got shape {x.shape}")

    # NumPy array
    if isinstance(x, np.ndarray):
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN, Inf, or -Inf values")
    # Torch tensor
    elif isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN, Inf, or -Inf values")
    else:
        raise TypeError(f"x must be np.ndarray or torch.Tensor, got {type(x)}")

def check_tabular(X):
    """
    Check that X is a valid tabular array/matrix.
    - Must be 2D
    - No NaN values
    - No ±Inf values
    - All entries are finite numbers
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (tabular), got shape {X.shape}")

    # NumPy array
    if isinstance(X, np.ndarray):
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN, Inf, or -Inf values")
    # Torch tensor
    elif isinstance(X, torch.Tensor):
        if not torch.isfinite(X).all():
            raise ValueError("X contains NaN, Inf, or -Inf values")
    else:
        raise TypeError(f"X must be np.ndarray or torch.Tensor, got {type(X)}")

def check_cover(cover):
    """Ensure cover values are binary (0 or 1)."""
    # Convert pandas Series to NumPy array
    if isinstance(cover, pd.Series) or isinstance(cover, pd.DataFrame):
        cover = cover.values

    if isinstance(cover, torch.Tensor):
        if cover.ndim != 1:
            raise ValueError(f"cover must be 1D, got {cover.shape}")
        if not torch.all((cover == 0) | (cover == 1)):
            raise ValueError("cover values must be 0 or 1.")
    elif isinstance(cover, (list, np.ndarray)):
        cover = np.array(cover)
        if cover.ndim != 1:
            raise ValueError(f"cover must be 1D, got {cover.shape}")
        if not np.all(np.isin(cover, [0, 1])):
            raise ValueError("cover values must be 0 or 1.")
    else:
        raise TypeError("cover must be numpy array, list, pandas Series, or torch tensor.")

def check_emptyness(array):
    """
    Check if the input array or tensor is empty.
    
    Parameters:
        array: Can be a list, numpy array, or torch tensor.
    
    Raises:
        ValueError: If the input array is empty.
    """
    # For PyTorch tensors
    if isinstance(array, torch.Tensor):
        if array.numel() == 0:
            raise ValueError("The tensor is empty.")
    
    # For NumPy arrays
    elif isinstance(array, np.ndarray):
        if array.size == 0:
            raise ValueError("The numpy array is empty.")
            
    else:
        raise TypeError("Input must be a list, tuple, numpy array, or torch tensor.")

    return True  # Optional: Return True if not empty

def check_groups(groups):
    """Ensure group labels are integers."""
    if isinstance(groups, torch.Tensor):
        if groups.ndim != 1:
            raise ValueError(f"groups must be 1D, got {groups.shape}")
        if not groups.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("Group labels must be integers (torch int type).")
    elif isinstance(groups, (list, np.ndarray)):
        groups = np.array(groups)
        if groups.ndim != 1:
            raise ValueError(f"groups must be 1D, got {groups.shape}")
        if not np.issubdtype(groups.dtype, np.integer):
            raise TypeError("group labels must be integers.")
    else:
        raise TypeError("groups must be numpy array, list, or torch tensor.")

def check_consistency(cover, groups):
    """Ensure cover and groups have the same length and type."""
    # Convert pandas Series to NumPy arrays
    if isinstance(cover, pd.Series) or isinstance(cover, pd.DataFrame):
        cover = cover.values
    if isinstance(groups, pd.Series) or isinstance(groups, pd.DataFrame):
        groups = groups.values

    # Check type
    if type(cover) is not type(groups):
        raise TypeError(f"cover and groups must be of the same type, got {type(cover)} and {type(groups)}")
    
    # Check length
    if len(cover) != len(groups):
        raise ValueError(
            f"cover and groups must have the same length, got cover of shape {getattr(cover, 'shape', len(cover))} "
            f"and groups of shape {getattr(groups, 'shape', len(groups))}"
        )
    
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
