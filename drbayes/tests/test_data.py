import pytest

from numpy.testing import assert_almost_equal
import numpy as np

from bayesian_benchmarks.data import regression_datasets, get_regression_data, get_classification_data

@pytest.mark.parametrize('d', ['boston'])
def test_regression(d):
    data = get_regression_data(d)

    assert_almost_equal(np.average(np.concatenate([data.X_train, data.X_test], 0), 0),
                        np.zeros(data.X_train.shape[1]))

    assert_almost_equal(np.std(np.concatenate([data.X_train, data.X_test], 0), 0),
                        np.ones(data.X_train.shape[1]),
                        decimal=3)

    assert_almost_equal(np.average(np.concatenate([data.Y_train, data.Y_test], 0), 0),
                        np.zeros(data.Y_train.shape[1]))

    assert_almost_equal(np.std(np.concatenate([data.Y_train, data.Y_test], 0), 0),
                        np.ones(data.Y_train.shape[1]),
                        decimal=3)

    assert data.X_train.shape[0] == data.Y_train.shape[0]
    assert data.X_test.shape[0] == data.Y_test.shape[0]
    assert data.X_train.shape[1] == data.X_test.shape[1]
    assert data.Y_train.shape[1] == data.Y_test.shape[1]

@pytest.mark.parametrize('d', ['iris', 'thyroid'])
def test_classification(d):
    data = get_classification_data(d)

    assert_almost_equal(np.average(np.concatenate([data.X_train, data.X_test], 0), 0),
                        np.zeros(data.X_train.shape[1]))

    assert_almost_equal(np.std(np.concatenate([data.X_train, data.X_test], 0), 0),
                        np.ones(data.X_train.shape[1]),
                        decimal=3)

    K = len(list(set(np.concatenate([data.Y_train, data.Y_test], 0).astype(int).flatten())))

    assert K == data.K

    assert data.X_train.shape[0] == data.Y_train.shape[0]
    assert data.X_test.shape[0] == data.Y_test.shape[0]
    assert data.X_train.shape[1] == data.X_test.shape[1]
    assert data.Y_train.shape[1] == data.Y_test.shape[1]
