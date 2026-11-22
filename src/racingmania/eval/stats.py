"""Statistics for the preregistered analysis: SWS, two-level bootstrap intervals, t-tests, Holm, IQM.

Local purpose: pure NumPy/SciPy functions on score matrices. A score matrix has one row per seed
(training run) and one column per track, holding the per-track success-weighted-speed score
``finished · T_ref / T`` (reports/preregistration.md, section 4). Nothing here reads files.

Global context: ``racingmania.report`` builds every table, figure, hypothesis test and website
number from the result files through these functions, exactly as preregistered (section 6 and the
amendments in reports/preregistration.md):
- 95 % intervals: two-level percentile bootstrap, 10,000 resamples, seeds resampled with
  replacement and tracks resampled with replacement; for a difference between two agents the same
  resampled tracks are used for both (paired on tracks). Resampling rows (seeds) and columns
  (tracks) independently is Owen's "pigeonhole bootstrap" for crossed designs (Owen 2007,
  Annals of Applied Statistics 1(2):386-411, https://arxiv.org/abs/0712.1111), whose variance
  estimate Owen shows to be mildly conservative (same paper, abstract and section 1).
- Confirmation: one-sided one-sample t-test (H1) and one-sided Welch t-test (H2, H3) on per-seed
  scores: Colas et al. 2019 (arXiv:1904.06979) recommend Welch's test, say the bootstrap test should
  never be used below N = 50 samples (section 4.5), report high false-positive rates below 5 seeds
  and find that about 10 seeds are needed to detect an effect of 2 pooled standard deviations with
  power 0.8 (section 4.3); the 2018 paper (arXiv:1806.08295) also recommends Welch's test, with
  Holm's step-down correction across the three hypotheses (Holm 1979, "A simple sequentially
  rejective multiple test procedure", Scandinavian Journal of Statistics 6(2):65-70,
  https://www.jstor.org/stable/4615733).
- Cross-checks: interquartile mean and probability of improvement with the definitions of Agarwal
  et al. 2021 (arXiv:2108.13264) as implemented in ``rliable/metrics.py``
  (https://github.com/google-research/rliable, Apache-2.0); the package itself is not imported
  because it fails to import with pandas 3 (preregistration amendment).
- Seed sensitivity: share of variance from seeds versus tracks, from the ANOVA (method-of-moments)
  estimators of a two-way crossed random-effects model without replication (Searle, Casella &
  McCulloch 1992, "Variance Components", Wiley, https://doi.org/10.1002/9780470316856, ch. 4).
  E[MS_seed] = σ²_res + T·σ²_seed and E[MS_track] = σ²_res + S·σ²_track give the estimators below.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy import stats as sps

N_BOOT = 10_000  # bootstrap resamples (preregistration, section 6)
CI_LEVEL = 95.0  # interval level in percent (preregistration, section 6)
BOOT_SEED = 0  # fixed generator seed so every rebuild of the report gives identical intervals


# ----------------------------------------------------------------------------- the metric
def track_scores(finished, ref_ticks, ticks) -> np.ndarray:
    """Per-track score ``finished · T_ref / T`` (0 when unfinished or without a reference).

    Same rule as ``eval/evaluator.py``; the report uses it to re-check the ``score`` column of
    every evaluation file.
    """
    f = np.asarray(finished, float)
    r = np.asarray(ref_ticks, float)
    t = np.asarray(ticks, float)
    ok = (f > 0) & (r > 0) & (t > 0)  # a score exists only for finished tracks with a reference time
    out = np.zeros(np.broadcast(f, r, t).shape)
    np.divide(r, t, out=out, where=ok)  # T_ref / T where defined, 0 elsewhere
    return out


def sws(scores) -> float:
    """Success-weighted speed of one run: the mean per-track score (NaN for no tracks)."""
    s = np.asarray(scores, float)
    return float(s.mean()) if s.size else math.nan


# ----------------------------------------------------------------------------- bootstrap
@dataclass
class Interval:
    """A point estimate with its percentile-bootstrap interval and the sample sizes behind it."""

    estimate: float
    lo: float
    hi: float
    n_seeds: int
    n_tracks: int
    n_boot: int = N_BOOT

    def as_dict(self) -> dict:
        return asdict(self)


def _nan_interval(n_seeds: int = 0, n_tracks: int = 0) -> Interval:
    return Interval(math.nan, math.nan, math.nan, n_seeds, n_tracks, 0)


@lru_cache(maxsize=8)
def _track_counts(n_tracks: int, n_boot: int, seed: int) -> np.ndarray:
    """How often each track is drawn in each bootstrap replicate, shape (n_boot, n_tracks).

    Drawing ``n`` items with replacement is one multinomial draw with equal probabilities, so the
    mean over a resample equals ``counts @ values / n``. Cached so that every agent scored on the
    same track set uses the same resampled tracks (common random numbers; this is what makes a
    difference "paired on tracks").
    """
    rng = np.random.default_rng([seed, 0])
    counts = rng.multinomial(n_tracks, np.full(n_tracks, 1.0 / n_tracks), size=n_boot).astype(np.float32)
    counts.setflags(write=False)  # shared through the cache, so it must not be modified
    return counts


def _seed_counts(n_seeds: int, n_boot: int, seed: int, stream: int) -> np.ndarray:
    """Resampling counts for the seeds of one agent (an independent stream per agent)."""
    rng = np.random.default_rng([seed, stream])
    return rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=n_boot).astype(np.float64)


def _replicate_means(M: np.ndarray, seed_counts: np.ndarray, track_counts: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Mean of ``M`` (seeds × tracks) in every replicate, given resampling counts for both levels.

    For replicate b the resampled mean is ``Σ_s Σ_t D[b,s] · C[b,t] · M[s,t] / (S·T)``, i.e.
    ``((D @ M) * C).sum(1) / (S·T)``; computed in chunks to bound memory.
    """
    S, T = M.shape
    out = np.empty(seed_counts.shape[0])
    for a in range(0, seed_counts.shape[0], chunk):
        b = min(a + chunk, seed_counts.shape[0])
        out[a:b] = ((seed_counts[a:b] @ M) * track_counts[a:b]).sum(1) / (S * T)
    return out


def _percentiles(reps: np.ndarray, level: float) -> tuple[float, float]:
    tail = (100.0 - level) / 2.0  # e.g. 2.5 and 97.5 for a 95 % interval
    lo, hi = np.percentile(reps, [tail, 100.0 - tail])
    return float(lo), float(hi)


def bootstrap_mean(M, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = CI_LEVEL) -> Interval:
    """Two-level percentile bootstrap interval of the mean of a seeds × tracks matrix.

    Each replicate resamples the seeds with replacement and, independently, the tracks with
    replacement (preregistration, section 6). With one seed only the tracks vary, so the interval
    then describes track sampling alone.
    """
    M = np.atleast_2d(np.asarray(M, float))
    S, T = M.shape
    if S == 0 or T == 0:
        return _nan_interval(S, T)
    reps = _replicate_means(M, _seed_counts(S, n_boot, seed, 1), _track_counts(T, n_boot, seed))
    lo, hi = _percentiles(reps, level)
    return Interval(float(M.mean()), lo, hi, S, T, n_boot)


def bootstrap_diff(A, B, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = CI_LEVEL) -> Interval:
    """Interval of mean(A) − mean(B) for two agents scored on the same tracks (same column order).

    Seeds of A and of B are resampled independently (they are different training runs); one track
    resample per replicate is shared by both, so the difference is paired on tracks
    (preregistration, section 6). ``n_seeds`` in the result is the smaller of the two groups.
    """
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A and B must have the same tracks (columns): {A.shape} vs {B.shape}")
    (Sa, T), Sb = A.shape, B.shape[0]
    if Sa == 0 or Sb == 0 or T == 0:
        return _nan_interval(min(Sa, Sb), T)
    C = _track_counts(T, n_boot, seed)
    reps = (_replicate_means(A, _seed_counts(Sa, n_boot, seed, 1), C)
            - _replicate_means(B, _seed_counts(Sb, n_boot, seed, 2), C))
    lo, hi = _percentiles(reps, level)
    return Interval(float(A.mean() - B.mean()), lo, hi, min(Sa, Sb), T, n_boot)


def bootstrap_set_gap(M1, M2, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = CI_LEVEL) -> Interval:
    """Interval of mean(M1) − mean(M2) for the same runs (rows) scored on two different track sets.

    Used for the train–test gap (Kirk et al. 2023, arXiv:2111.09794): row ``s`` of both matrices
    is the same seed, so seeds are resampled jointly, while each set's tracks are resampled on
    their own (the sets share no tracks).
    """
    M1 = np.atleast_2d(np.asarray(M1, float))
    M2 = np.atleast_2d(np.asarray(M2, float))
    if M1.shape[0] != M2.shape[0]:
        raise ValueError("both matrices need one row per seed, in the same order")
    S = M1.shape[0]
    if S == 0 or M1.shape[1] == 0 or M2.shape[1] == 0:
        return _nan_interval(S, 0)
    D = _seed_counts(S, n_boot, seed, 3)
    reps = (_replicate_means(M1, D, _track_counts(M1.shape[1], n_boot, seed))
            - _replicate_means(M2, D, _track_counts(M2.shape[1], n_boot, seed + 1)))
    lo, hi = _percentiles(reps, level)
    return Interval(float(M1.mean() - M2.mean()), lo, hi, S, M1.shape[1] + M2.shape[1], n_boot)


# ----------------------------------------------------------------------------- t-tests and Holm
@dataclass
class TTest:
    """Result of a one-sided t-test on per-seed scores."""

    statistic: float
    p: float
    df: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def one_sample_t(x, mu0: float, alternative: str = "greater") -> TTest:
    """One-sample t-test of mean(x) against ``mu0`` (H1: per-seed SWS against 1.0).

    ``scipy.stats.ttest_1samp`` (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html).
    Needs at least two seeds; otherwise every field is NaN.
    """
    x = np.asarray(x, float)
    if x.size < 2:
        return TTest(math.nan, math.nan, math.nan, int(x.size))
    with warnings.catch_warnings():  # zero variance gives an infinite or NaN statistic, which is reported as is
        warnings.simplefilter("ignore", RuntimeWarning)
        r = sps.ttest_1samp(x, popmean=mu0, alternative=alternative)
    return TTest(float(r.statistic), float(r.pvalue), float(x.size - 1), int(x.size))


def welch_t(a, b, delta: float = 0.0, alternative: str = "greater") -> TTest:
    """Welch's t-test that mean(a) − mean(b) exceeds ``delta`` (H2, H3 with delta = 0.10).

    Testing ``μa − μb > δ`` is the ordinary Welch test of ``a`` against ``b + δ``: shifting one
    sample moves its mean and leaves its variance unchanged. ``scipy.stats.ttest_ind`` with
    ``equal_var=False`` (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html);
    Colas et al. 2019 (arXiv:1904.06979) recommend Welch's test for comparing RL algorithms.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return TTest(math.nan, math.nan, math.nan, int(min(a.size, b.size)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = sps.ttest_ind(a, b + delta, equal_var=False, alternative=alternative)
    return TTest(float(r.statistic), float(r.pvalue), float(getattr(r, "df", math.nan)), int(min(a.size, b.size)))


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (Holm 1979).

    With the m p-values sorted ascending, ``p_adj(i) = max_{j ≤ i} min(1, (m − j + 1) · p(j))``.
    NaN entries (hypotheses that could not be tested yet) stay NaN but still count in ``m``,
    which is conservative: they behave like the largest p-values.
    """
    p = np.asarray(pvals, float)
    m = p.size
    adj = np.full(m, math.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok], kind="stable")]  # indices of the testable p-values, smallest first
    running = 0.0
    for rank, idx in enumerate(order):  # rank 0 gets factor m, rank 1 gets m − 1, ...
        running = max(running, min(1.0, (m - rank) * p[idx]))  # the max keeps the adjusted values monotone
        adj[idx] = running
    return adj


# ----------------------------------------------------------------------------- rliable cross-checks
def iqm(scores) -> float:
    """Interquartile mean over all runs and tasks: the 25 % trimmed mean (Agarwal et al. 2021).

    Same as ``rliable.metrics.aggregate_iqm``: ``scipy.stats.trim_mean(scores, 0.25, axis=None)``.
    """
    s = np.asarray(scores, float).ravel()
    return float(sps.trim_mean(s, proportiontocut=0.25)) if s.size else math.nan


def probability_of_improvement(X, Y) -> float:
    """P(X > Y) averaged over tasks (here: tracks), Agarwal et al. 2021, as in ``rliable``.

    ``X`` is runs_x × tasks and ``Y`` runs_y × tasks. Per task it is the Mann-Whitney U statistic
    divided by ``runs_x · runs_y``: the share of run pairs where X scores higher, ties counting
    one half. (rliable returns 0.5 for identical columns, which this formula also gives.)
    """
    X = np.atleast_2d(np.asarray(X, float))
    Y = np.atleast_2d(np.asarray(Y, float))
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same tasks (columns)")
    if X.size == 0 or Y.size == 0:
        return math.nan
    gt = (X[:, None, :] > Y[None, :, :]).mean(axis=(0, 1))  # share of pairs with x > y, per task
    eq = (X[:, None, :] == Y[None, :, :]).mean(axis=(0, 1))  # share of tied pairs, per task
    return float((gt + 0.5 * eq).mean())


# ----------------------------------------------------------------------------- seed sensitivity
def seed_spread(per_seed) -> dict:
    """Minimum, maximum, mean and standard deviation (ddof = 1) of per-seed scores."""
    x = np.asarray(per_seed, float)
    if x.size == 0:
        return {"n": 0, "mean": math.nan, "min": math.nan, "max": math.nan, "std": math.nan}
    return {"n": int(x.size), "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()),
            "std": float(x.std(ddof=1)) if x.size > 1 else math.nan}


def variance_components(M) -> dict:
    """Shares of variance from seeds, tracks and their interaction in a seeds × tracks matrix.

    Two-way crossed random-effects model without replication, ``y_st = μ + a_s + b_t + e_st``,
    with ANOVA estimators: ``σ²_seed = (MS_seed − MS_res)/T``, ``σ²_track = (MS_track − MS_res)/S``,
    ``σ²_res = MS_res``, negative estimates truncated at 0 (Searle, Casella & McCulloch 1992).
    The simulator is deterministic, so the residual is the seed × track interaction (a seed that
    is good on some tracks and bad on others). Needs at least 2 seeds and 2 tracks.
    """
    M = np.atleast_2d(np.asarray(M, float))
    S, T = M.shape
    nan = {"var_seed": math.nan, "var_track": math.nan, "var_interaction": math.nan,
           "share_seed": math.nan, "share_track": math.nan, "share_interaction": math.nan, "n_seeds": S, "n_tracks": T}
    if S < 2 or T < 2:
        return nan
    grand = M.mean()
    seed_means, track_means = M.mean(1), M.mean(0)
    ms_seed = T * ((seed_means - grand) ** 2).sum() / (S - 1)  # between-seed mean square
    ms_track = S * ((track_means - grand) ** 2).sum() / (T - 1)  # between-track mean square
    resid = M - seed_means[:, None] - track_means[None, :] + grand  # interaction residuals
    ms_res = (resid ** 2).sum() / ((S - 1) * (T - 1))
    v_seed = max(0.0, (ms_seed - ms_res) / T)
    v_track = max(0.0, (ms_track - ms_res) / S)
    v_res = float(ms_res)
    total = v_seed + v_track + v_res
    if total <= 0:
        return {**nan, "var_seed": 0.0, "var_track": 0.0, "var_interaction": 0.0}
    return {"var_seed": float(v_seed), "var_track": float(v_track), "var_interaction": v_res,
            "share_seed": float(v_seed / total), "share_track": float(v_track / total),
            "share_interaction": float(v_res / total), "n_seeds": S, "n_tracks": T}
