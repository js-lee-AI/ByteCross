import itertools

import numpy as np

from bytecross.campaign import block_interval
from bytecross.crossing import crossing, crossing_interval
from bytecross.predict import predict
from bytecross.traffic import MODELS, crossover

K = 1024


def test_table12_means():
    # mean differences of Tables 12 and 13 give the measured crossings of Table 2
    cases = [
        (range(20, 53, 4), [-0.328, -0.009, 0.025, 0.212, 0.608, 0.638, 0.781, 1.125, 1.418], 25.1),
        (range(36, 81, 4), [-0.666, -0.356, -0.245, -0.021, 0.014, 0.670, 0.453, 0.882, 0.913, 1.384, 1.511,
                            1.839], 50.4),
        (range(56, 129, 8), [-0.169, -0.199, 0.170, 0.476, 0.954, 1.161, 1.398, 1.880, 1.966, 2.659], 68.3),
        (range(4, 33, 4), [-0.498, -0.402, -0.145, -0.019, 0.137, 0.407, 0.562, 0.756], 16.5),
        (range(28, 53, 4), [-1.529, -1.131, -0.562, 0.115, 0.562, 1.095, 1.568], 39.3),
        (range(56, 85, 4), [-1.416, -1.191, -0.681, -0.186, 0.932, 1.222, 1.707, 1.901], 68.7),
    ]
    for ks, diffs, expected in cases:
        assert round(crossing([k * K for k in ks], diffs) / K, 1) == expected


def test_crossing_interpolation():
    assert crossing([10, 20, 30], [-1.0, -0.5, 1.5]) == 22.5
    assert crossing([10, 20, 30], [-1.0, 0.0, 1.0]) == 20.0
    assert crossing([10, 20, 30], [0.5, 1.0, 2.0]) is None
    assert crossing([10, 20, 30], [-2.0, -1.0, -0.5]) is None


def test_resample_interval():
    # four blocks whose differences are linear in context with crossings at 13K..16K, so the
    # crossing of any resample is the mean of its block crossings
    ctxs = [8 * K, 12 * K, 16 * K, 20 * K]
    roots = np.array([13, 14, 15, 16]) * K
    diffs = 1e-4 * (np.array(ctxs)[None, :] - roots[:, None])
    assert abs(crossing(ctxs, diffs.mean(axis=0)) - roots.mean()) < 1e-6
    ci, hits, total = crossing_interval(ctxs, diffs)
    assert (hits, total) == (256, 256)
    means = [roots[list(r)].mean() for r in itertools.product(range(4), repeat=4)]
    assert np.allclose(ci, np.quantile(means, [0.025, 0.975]))
    assert ci[0] < roots.mean() < ci[1]
    # identical blocks: the interval collapses onto the point estimate
    ci, _, _ = crossing_interval(ctxs, np.repeat(diffs[:1], 4, axis=0))
    assert np.allclose(ci, roots[0])


def test_block_interval():
    v = [1.0, 2.0, 4.0]
    means = [np.mean([v[i] for i in r]) for r in itertools.product(range(3), repeat=3)]
    assert np.allclose(block_interval(v), np.quantile(means, [0.025, 0.975]))


def test_prediction_without_kernel_costs():
    # with no kernel cost beyond the byte time, the latency crossing is the byte crossover
    m = MODELS["llama-3.1-8b"]
    x = predict(m, lambda n: 8 + n / 20000, lambda n: 0.0, lambda n: 0.0, 0.5, 0.3)
    assert abs(x - crossover(m, 0.5, 0.3)) < 1
    # a larger kernel cost of the projection branch moves the crossing earlier
    assert predict(m, lambda n: 8 + n / 20000, lambda n: 1.0, lambda n: 0.0, 0.5, 0.3) < x


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
