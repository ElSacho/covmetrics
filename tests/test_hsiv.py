import numpy as np
import torch
import pytest
from conditional_coverage_metrics import *

def to_backend(array, backend, dtype="float"):
    """Helper to create numpy/torch/list arrays with correct types."""
    if backend == "numpy":
        return np.array(array, dtype=float if dtype == "float" else int)
    elif backend == "torch":
        return torch.tensor(array, dtype=torch.float32 if dtype == "float" else torch.int64)
    else:
        raise ValueError("Unsupported backend")


# -------------------- CORE TESTS --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_independent_inputs_small_value(backend):
    """If cover and sizes are independent, HSIC should be close to 0."""
    rng = np.random.default_rng(0)
    cover = rng.integers(0, 2, size=50)   # binary cover
    sizes = rng.normal(size=50)           # random noise, independent

    cover_b = to_backend(cover, backend)
    sizes_b = to_backend(sizes, backend)

    estimator = HSIC(sigma_x=1, sigma_y=1)
    val = estimator.evaluate(cover_b, sizes_b)
    assert isinstance(val, float)
    assert val >= 0.0
    assert val < 1.0   # should be small-ish for independent


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_dependent_inputs_large_value(backend):
    """If cover is correlated with sizes, HSIC should be larger."""
    cover = [0]*25 + [1]*25
    sizes = [1]*25 + [10]*25   # strongly dependent

    cover_b = to_backend(cover, backend, dtype="float")
    sizes_b = to_backend(sizes, backend, dtype="float")

    estimator = HSIC(sigma_x=1, sigma_y=1)
    val = estimator.evaluate(cover_b, sizes_b)
    assert isinstance(val, float)
    assert val > 0.1   # must detect dependence


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_return_type_is_float(backend):
    cover = to_backend([0, 1, 0, 1], backend)
    sizes = to_backend([1.0, 2.0, 3.0, 4.0], backend)
    estimator = HSIC()
    val = estimator.evaluate(cover, sizes)
    assert isinstance(val, float)


# -------------------- BACKEND CONSISTENCY --------------------

def test_numpy_and_torch_equivalence():
    rng = np.random.default_rng(0)
    cover = rng.integers(0, 2, size=30)
    sizes = rng.normal(size=30)

    estimator = HSIC()

    val_numpy = estimator.evaluate(np.array(cover, dtype=float),
                                   np.array(sizes, dtype=float))

    val_torch = estimator.evaluate(torch.tensor(cover, dtype=torch.float32),
                                   torch.tensor(sizes, dtype=torch.float32))

    # should be very close
    assert np.isclose(val_numpy, val_torch, atol=1e-6)


# -------------------- ERROR HANDLING --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_length_mismatch_raises(backend):
    cover = to_backend([0, 1, 1], backend)
    sizes = to_backend([1.0, 2.0], backend)  # shorter
    estimator = HSIC()
    with pytest.raises((ValueError, IndexError)):
        estimator.evaluate(cover, sizes)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_empty_inputs_raise(backend):
    cover = to_backend([], backend)
    sizes = to_backend([], backend)
    estimator = HSIC()
    with pytest.raises((ValueError, AssertionError)):
        estimator.evaluate(cover, sizes)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_invalid_cover_values_raise(backend):
    """Cover must be 0/1 only."""
    estimator = HSIC()
    sizes = to_backend([1.0, 2.0, 3.0, 4.0], backend)

    cover_bad = to_backend([0, 2, 1, -1], backend)
    with pytest.raises((ValueError, AssertionError)):
        estimator.evaluate(cover_bad, sizes)

    cover_float_bad = to_backend([0.0, 0.5, 1.0, 0.2], backend)
    with pytest.raises((ValueError, AssertionError)):
        estimator.evaluate(cover_float_bad, sizes)


def test_backend_mismatch_numpy_cover_torch_sizes():
    cover = np.array([0, 1, 0, 1], dtype=float)            # numpy
    sizes = torch.tensor([1.0, 2.0, 3.0, 4.0])             # torch
    estimator = HSIC()
    val = estimator.evaluate(cover, sizes)
    assert isinstance(val, float)


def test_backend_mismatch_torch_cover_numpy_sizes():
    cover = torch.tensor([0.0, 1.0, 0.0, 1.0])
    sizes = np.array([1.0, 2.0, 3.0, 4.0])
    estimator = HSIC()
    val = estimator.evaluate(cover, sizes)
    assert isinstance(val, float)


# -------------------- RANDOMIZED TEST --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_random_binary_cover_runs(backend):
    """Just checks that HSIC runs and returns nonnegative for random data."""
    rng = np.random.default_rng(123)
    cover = rng.integers(0, 2, size=100)
    sizes = rng.normal(size=100)

    if backend == "torch":
        cover = torch.tensor(cover, dtype=torch.float32)
        sizes = torch.tensor(sizes, dtype=torch.float32)

    estimator = HSIC(sigma_x=1, sigma_y=1)
    val = estimator.evaluate(cover, sizes)
    assert isinstance(val, float)
    assert val >= 0.0
