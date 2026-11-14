"""Unit tests for ``racingmania.eval.stats`` against hand-computed cases.

Global context: every reported number and hypothesis decision passes through these functions
(reports/preregistration.md, sections 4-6), so each is checked on a case small enough to compute
by hand, plus a synthetic coverage check of the two-level bootstrap.
"""

import math

import numpy as np
import pytest

from racingmania.eval import stats


def test_sws_of_three_tracks():
    # scores: 100/80 = 1.25, 200/250 = 0.8, unfinished = 0  ->  SWS = 2.05 / 3
    s = stats.track_scores([1, 1, 0], [100, 200, 300], [80, 250, 900])
    assert s == pytest.approx([1.25, 0.8, 0.0])
    assert stats.sws(s) == pytest.approx(2.05 / 3)


def test_track_scores_zero_without_reference():
    assert stats.track_scores([1], [0], [100]).tolist() == [0.0]
    assert math.isnan(stats.sws([]))


def test_holm_known_values():
    # sorted: 0.005*4 = 0.02, 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> raised to 0.06 (monotone)
    adj = stats.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_keeps_nan_conservative():
    assert stats.holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])
    adj = stats.holm([0.01, math.nan, 0.04])  # m stays 3: 0.01*3 = 0.03, 0.04*2 = 0.08
    assert adj[0] == pytest.approx(0.03) and math.isnan(adj[1]) and adj[2] == pytest.approx(0.08)


def test_one_sample_t_matches_closed_form_df2():
    # x = 1.1, 1.2, 1.3: t = 0.2 / (0.1/sqrt(3)) = 2*sqrt(3); for df = 2, sf(t) = (1 - t/sqrt(t^2 + 2)) / 2
    r = stats.one_sample_t([1.1, 1.2, 1.3], 1.0)
    t = 2 * math.sqrt(3)
    assert r.statistic == pytest.approx(t)
    assert r.df == 2
    assert r.p == pytest.approx(0.5 * (1 - t / math.sqrt(t * t + 2)))


def test_welch_t_with_delta_matches_closed_form():
    # a = 1,2,3 ; b = 0,0,0 ; delta 0.1: t = (2 - 0.1) / sqrt(1/3), Welch df = 2
    r = stats.welch_t([1, 2, 3], [0, 0, 0], delta=0.1)
    t = 1.9 / math.sqrt(1 / 3)
    assert r.statistic == pytest.approx(t)
    assert r.df == pytest.approx(2.0)
    assert r.p == pytest.approx(0.5 * (1 - t / math.sqrt(t * t + 2)))


def test_t_tests_need_two_seeds():
    assert math.isnan(stats.one_sample_t([1.2], 1.0).p)
    assert math.isnan(stats.welch_t([1.2, 1.3], [0.4], 0.1).p)


def test_iqm_is_25_percent_trimmed_mean():
    assert stats.iqm(np.arange(1, 9).reshape(2, 4)) == pytest.approx(4.5)  # middle values 3,4,5,6


def test_probability_of_improvement_hand_case():
    # task 1: x = 1,3 vs y = 2,2 -> 2 of 4 pairs -> 0.5 ; task 2: x = 1,1 vs y = 1,0 -> (0.5+1+0.5+1)/4 = 0.75
    X = np.array([[1, 1], [3, 1]])
    Y = np.array([[2, 1], [2, 0]])
    assert stats.probability_of_improvement(X, Y) == pytest.approx(0.625)
    assert stats.probability_of_improvement(X, X) == pytest.approx(0.5)


def test_variance_components_additive_case():
    # seed effects -1, +1 (variance 2), track effects -2, 0, 2 (variance 4), no interaction
    M = np.array([-1.0, 1.0])[:, None] + np.array([-2.0, 0.0, 2.0])[None, :]
    v = stats.variance_components(M)
    assert v["var_seed"] == pytest.approx(2.0) and v["var_track"] == pytest.approx(4.0)
    assert v["var_interaction"] == pytest.approx(0.0, abs=1e-12)
    assert v["share_seed"] == pytest.approx(1 / 3) and v["share_track"] == pytest.approx(2 / 3)
    assert math.isnan(stats.variance_components(M[:1])["share_seed"])


def test_seed_spread():
    s = stats.seed_spread([0.5, 0.7, 0.9])
    assert (s["min"], s["max"], s["n"]) == (0.5, 0.9, 3)
    assert s["std"] == pytest.approx(0.2)


def test_bootstrap_is_seeded_and_contains_estimate():
    rng = np.random.default_rng(1)
    M = rng.uniform(0, 1.2, size=(4, 50))
    a = stats.bootstrap_mean(M, n_boot=2000)
    b = stats.bootstrap_mean(M, n_boot=2000)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    assert a.lo < a.estimate < a.hi and a.estimate == pytest.approx(M.mean())
    assert (a.n_seeds, a.n_tracks) == (4, 50)


def test_bootstrap_ci_covers_true_mean_in_synthetic_check():
    """Crossed design y = mu + seed + track + noise: the 95 % interval should cover mu most of the time."""
    rng = np.random.default_rng(123)
    mu, hits, trials = 0.8, 0, 60
    for _ in range(trials):
        M = (mu + rng.normal(0, 0.05, (10, 1)) + rng.normal(0, 0.2, (1, 150)) + rng.normal(0, 0.05, (10, 150)))
        ci = stats.bootstrap_mean(M, n_boot=1000, seed=int(rng.integers(1 << 30)))
        hits += ci.lo <= mu <= ci.hi
    assert hits / trials >= 0.85  # nominal 0.95; the pigeonhole bootstrap is mildly conservative


def test_bootstrap_diff_is_paired_on_tracks():
    rng = np.random.default_rng(2)
    base = rng.uniform(0, 1, size=(1, 80))
    A, B = base + 0.3, base  # one seed each, identical track effects: the paired difference is exactly 0.3
    d = stats.bootstrap_diff(A, B, n_boot=1000)
    assert d.estimate == pytest.approx(0.3)
    assert d.lo == pytest.approx(0.3) and d.hi == pytest.approx(0.3)
    with pytest.raises(ValueError):
        stats.bootstrap_diff(A, B[:, :10])


def test_bootstrap_set_gap():
    M1 = np.full((2, 5), 0.9)
    M2 = np.full((2, 7), 0.6)
    g = stats.bootstrap_set_gap(M1, M2, n_boot=500)
    assert g.estimate == pytest.approx(0.3) and g.lo == pytest.approx(0.3) and g.hi == pytest.approx(0.3)


def test_empty_inputs_give_nan():
    assert math.isnan(stats.bootstrap_mean(np.zeros((0, 3))).estimate)
    assert math.isnan(stats.iqm([]))
