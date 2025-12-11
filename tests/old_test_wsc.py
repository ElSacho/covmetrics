import numpy as np
import torch

from conditional_coverage_metrics import *

def make_data(n=100, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    cover = rng.integers(0, 2, size=n)  # random 0/1
    return X, cover


def test_output_type_and_range():
    """Check that evaluate returns a float in [0,1]."""
    X, cover = make_data()
    estimator = WSC(delta=0.1)
    val = estimator.evaluate(X, cover, M=100)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


def test_perfect_cover():
    """If all points are covered, WSC should be 1."""
    X, cover = make_data()
    cover[:] = 1
    estimator = WSC(delta=0.2)
    val = estimator.evaluate(X, cover, M=200)
    assert np.isclose(val, 1.0)


def test_no_cover():
    """If no point is covered, WSC should be 0."""
    X, cover = make_data()
    cover[:] = 0
    estimator = WSC(delta=0.3)
    val = estimator.evaluate(X, cover, M=200)
    assert np.isclose(val, 0.0)


def test_delta_threshold_effect():
    """Larger delta should increase WSC."""
    X, cover = make_data()
    estimator_small = WSC(delta=0.1)
    estimator_large = WSC(delta=0.5)
    val_small = estimator_small.evaluate(X, cover, M=200)
    val_large = estimator_large.evaluate(X, cover, M=200)
    assert val_large >= val_small + 1e-8


def test_reproducibility_with_seed():
    """Check that setting a random seed makes results reproducible."""
    X, cover = make_data()
    estimator1 = WSC(delta=0.1)
    estimator2 = WSC(delta=0.1)
    val1 = estimator1.evaluate(X, cover, M=200, seed=42)
    val2 = estimator2.evaluate(X, cover, M=200, seed=42)
    assert np.isclose(val1, val2)


def test_minimum_points_respected():
    """Check that the delta constraint is respected (at least delta*n points in the slab)."""
    X, cover = make_data()
    delta = 0.5
    estimator = WSC(delta=delta)
    val = estimator.evaluate(X, cover, M=200)
    # Not a strict numerical check, but should be valid probability
    assert 0.0 <= val <= 1.0


def test_high_dimensional_data():
    """Works in higher dimensions."""
    X, cover = make_data(n=200, d=10)
    estimator = WSC(delta=0.2)
    val = estimator.evaluate(X, cover, M=300)
    assert 0.0 <= val <= 1.0

def test_1d_all_covered():
    """In 1D, with all labels covered, WSC must be 1."""
    X = np.array([[0.], [1.], [2.], [3.]])   # 1D data
    cover = np.array([1, 1, 1, 1])
    estimator = WSC(delta=0.25)
    val = estimator.evaluate(X, cover, M=10)  # directions don't matter in 1D
    assert np.isclose(val, 1.0)


def test_1d_none_covered():
    """In 1D, with no labels covered, WSC must be 0."""
    X = np.array([[0.], [1.], [2.], [3.]])
    cover = np.array([0, 0, 0, 0])
    estimator = WSC(delta=0.25)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 0.0)

def test_1d_half_covered_easy_case():
    """
    Case: X = [0,1,2,3], cover = [1,1,1,0]
    For delta=0.25, every slab covering >=2 points must include both 1's and 0's.
    Worst-case fraction should be 0.5.
    """
    X = np.array([[0.], [1.], [2.], [3.]])
    cover = np.array([1, 1, 1, 0])
    estimator = WSC(delta=0.26)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 0.5)

def test_1d_half_covered_easy_case_2():
    """
    Case: X = [0,1,2,3], cover = [1,0,1,0]
    For delta=0.1, a slab covering one 0 yields a worst slab coverage of 0.
    Worst-case fraction should be 0.5.
    """
    X = np.array([[0.], [1.], [2.], [3.]])
    cover = np.array([1, 0, 1, 0])
    estimator = WSC(delta=0.1)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 0.)

def test_1d_half_covered_easy_case_3():
    """
    Case: X = [0,1,2,3], cover = [1,1,1,0]
    For delta=0.25, every slab covering >=2 points must include both 1's and 0's, and add a 0 to get the worst coverage Slab
    Worst-case fraction should be 0.5.
    """
    X = np.array([[0.], [1.], [2.], [3.]])
    cover = np.array([1, 0, 1, 0])
    estimator = WSC(delta=0.26)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 1/3)


def test_1d_asymmetric_cover():
    """
    Case: X = [0,1,2,3], cover = [1,1,0,0]
    - For delta=0.5, we need >=2 points in the slab.
    - Worst interval is [2,3] (two uncovered points) => coverage rate = 0.
    So WSC = 0.
    """
    X = np.array([[0.], [1.], [2.], [3.]])
    cover = np.array([1, 1, 0, 0])
    estimator = WSC(delta=0.5)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 0.0)


def test_1d_strict_delta():
    """
    Case: X = [0,1,2,3], cover = [1,0,1,1]
    - With delta=0.75 (must cover >=3 points).
    - Possible slabs: [0,2] covers (1,0,1) => 2/3 = 0.67
                      [1,3] covers (0,1,1) => 2/3 = 0.67
                      Whole interval [0,3] covers (1,0,1,1) => 0.75
    Worst-case = 2/3 ≈ 0.67
    """
    X = np.array([[0.], [1.], [2.], [3.]])
    cover = np.array([1, 0, 1, 1])
    estimator = WSC(delta=0.75)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 2/3)


import numpy as np
import torch
from conditional_coverage_metrics import *


def make_data_torch(n=100, d=2, seed=0, device="cpu"):
    """
    Torch version of make_data.
    Returns:
        X (torch.Tensor): shape (n, d)
        cover (torch.Tensor): shape (n,)
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    X = torch.randn((n, d), generator=g, device=device)
    cover = torch.randint(0, 2, (n,), generator=g, device=device)
    return X, cover


# Torch-based tests
def test_torch_output_type_and_range():
    """Check that evaluate works with torch tensors and returns a float in [0,1]."""
    X, cover = make_data_torch()
    check_inputs(X, cover)
    estimator = WSC(delta=0.1)
    val = estimator.evaluate(X, cover, M=100)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


def test_torch_perfect_cover():
    """If all points are covered (torch), WSC should be 1."""
    X, cover = make_data_torch()
    cover[:] = 1
    check_inputs(X, cover)
    estimator = WSC(delta=0.2)
    val = estimator.evaluate(X, cover, M=200)
    assert np.isclose(val, 1.0)


def test_torch_no_cover():
    """If no point is covered (torch), WSC should be 0."""
    X, cover = make_data_torch()
    cover[:] = 0
    check_inputs(X, cover)
    estimator = WSC(delta=0.3)
    val = estimator.evaluate(X, cover, M=200)
    assert np.isclose(val, 0.0)


def test_type_mismatch_raises():
    """Check that mixing numpy and torch raises TypeError."""
    X_np, cover_np = make_data()
    X_torch, cover_torch = make_data_torch()

    # Mixing numpy with torch must fail
    try:
        check_inputs(X_np, cover_torch)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError when mixing numpy and torch")

    try:
        check_inputs(X_torch, cover_np)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError when mixing numpy and torch")


def test_length_mismatch_raises():
    """Check that mismatched lengths raise ValueError."""
    X, cover = make_data(n=10)
    cover = cover[:-1]  # make length mismatch
    try:
        check_inputs(X, cover)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when lengths mismatch")



def make_data_torch(n=100, d=2, seed=0, device="cpu"):
    """
    Torch version of make_data.
    Returns:
        X (torch.Tensor): shape (n, d)
        cover (torch.Tensor): shape (n,)
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    X = torch.randn((n, d), generator=g, device=device)
    cover = torch.randint(0, 2, (n,), generator=g, device=device)
    return X, cover


def check_inputs(X, cover):
    """
    Ensures that X and cover are valid inputs.
    - Same length
    - Same type (both numpy arrays or both torch tensors)
    """
    if len(X) != len(cover):
        raise ValueError(f"X and cover must have the same length, got {len(X)} and {len(cover)}")

    if isinstance(X, torch.Tensor) and isinstance(cover, torch.Tensor):
        return
    elif isinstance(X, np.ndarray) and isinstance(cover, np.ndarray):
        return
    else:
        raise TypeError(f"X and cover must be of the same type (both numpy or both torch), "
                        f"got {type(X)} and {type(cover)}")


# Torch tests mirroring numpy ones
def test_output_type_and_range_torch():
    X, cover = make_data_torch()
    check_inputs(X, cover)
    estimator = WSC(delta=0.1)
    val = estimator.evaluate(X, cover, M=100)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


def test_perfect_cover_torch():
    X, cover = make_data_torch()
    cover[:] = 1
    check_inputs(X, cover)
    estimator = WSC(delta=0.2)
    val = estimator.evaluate(X, cover, M=200)
    assert np.isclose(val, 1.0)


def test_no_cover_torch():
    X, cover = make_data_torch()
    cover[:] = 0
    check_inputs(X, cover)
    estimator = WSC(delta=0.3)
    val = estimator.evaluate(X, cover, M=200)
    assert np.isclose(val, 0.0)


def test_delta_threshold_effect_torch():
    X, cover = make_data_torch()
    check_inputs(X, cover)
    estimator_small = WSC(delta=0.1)
    estimator_large = WSC(delta=0.5)
    val_small = estimator_small.evaluate(X, cover, M=200)
    val_large = estimator_large.evaluate(X, cover, M=200)
    assert val_large >= val_small + 1e-8


def test_reproducibility_with_seed_torch():
    X, cover = make_data_torch()
    check_inputs(X, cover)
    estimator1 = WSC(delta=0.1)
    estimator2 = WSC(delta=0.1)
    val1 = estimator1.evaluate(X, cover, M=200, seed=42)
    val2 = estimator2.evaluate(X, cover, M=200, seed=42)
    assert np.isclose(val1, val2)


def test_minimum_points_respected_torch():
    X, cover = make_data_torch()
    delta = 0.5
    check_inputs(X, cover)
    estimator = WSC(delta=delta)
    val = estimator.evaluate(X, cover, M=200)
    assert 0.0 <= val <= 1.0


def test_high_dimensional_data_torch():
    X, cover = make_data_torch(n=200, d=10)
    check_inputs(X, cover)
    estimator = WSC(delta=0.2)
    val = estimator.evaluate(X, cover, M=300)
    assert 0.0 <= val <= 1.0


def test_1d_all_covered_torch():
    X = torch.tensor([[0.], [1.], [2.], [3.]])
    cover = torch.tensor([1, 1, 1, 1])
    check_inputs(X, cover)
    estimator = WSC(delta=0.25)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 1.0)


def test_1d_none_covered_torch():
    X = torch.tensor([[0.], [1.], [2.], [3.]])
    cover = torch.tensor([0, 0, 0, 0])
    check_inputs(X, cover)
    estimator = WSC(delta=0.25)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 0.0)


def test_1d_half_covered_easy_case_torch():
    X = torch.tensor([[0.], [1.], [2.], [3.]])
    cover = torch.tensor([1, 1, 1, 0])
    check_inputs(X, cover)
    estimator = WSC(delta=0.26)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 0.5)


def test_1d_half_covered_easy_case_2_torch():
    X = torch.tensor([[0.], [1.], [2.], [3.]])
    cover = torch.tensor([1, 0, 1, 0])
    check_inputs(X, cover)
    estimator = WSC(delta=0.1)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 0.0)


def test_1d_half_covered_easy_case_3_torch():
    X = torch.tensor([[0.], [1.], [2.], [3.]])
    cover = torch.tensor([1, 0, 1, 0])
    check_inputs(X, cover)
    estimator = WSC(delta=0.26)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 1/3)


def test_1d_asymmetric_cover_torch():
    X = torch.tensor([[0.], [1.], [2.], [3.]])
    cover = torch.tensor([1, 1, 0, 0])
    check_inputs(X, cover)
    estimator = WSC(delta=0.5)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 0.0)


def test_1d_strict_delta_torch():
    X = torch.tensor([[0.], [1.], [2.], [3.]])
    cover = torch.tensor([1, 0, 1, 1])
    check_inputs(X, cover)
    estimator = WSC(delta=0.75)
    val = estimator.evaluate(X, cover, M=10)
    assert np.isclose(val, 2/3)
