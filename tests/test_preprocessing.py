import numpy as np
import pytest
from scipy.signal import detrend
from eyeprep.preprocessing import remove_blinks, to_dva, interpolate, normalize_pupil, moving_average,  gaussian_smooth, downsample, drift_correct, make_mask   # NEW functions
from pathlib import Path



ref = np.load(Path(__file__).parent / "reference" / "sacloc_run01.npz")



@pytest.mark.parametrize("method, key", [("pupil_off", "blinks"), ("pupil_off_smooth", "blinks_smooth")])
def test_blinks(method, key):
    np.testing.assert_allclose(remove_blinks(ref["raw"].copy(), method=method, sampling_rate=1000),
                               ref[key], equal_nan=True)

def test_blinks_unknown_method():
    with pytest.raises(ValueError):
        remove_blinks(ref["raw"], method="nope")

def test_dva():
    np.testing.assert_allclose(to_dva(ref["blinks"].copy(), [960, 540], 52.0), ref["dva"], equal_nan=True)

def test_interpolation():
    np.testing.assert_allclose(interpolate(ref["dva"][:, 1].copy()), ref["x_int"])

def test_normalize_pupil():
    np.testing.assert_allclose(normalize_pupil(ref["p_int"]), ref["p_norm"])

def test_normalize_pupil_range():                      # known-answer test
    out = normalize_pupil(np.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])

@pytest.mark.parametrize("axis", ["x", "y"])
def test_moving_average(axis):
    np.testing.assert_allclose(moving_average(ref[f"{axis}_int"], 1000, 50),
                               ref[f"{axis}_smooth"], equal_nan=True)

@pytest.mark.parametrize("axis", ["x", "y"])
def test_gaussian_smooth(axis):
    np.testing.assert_allclose(gaussian_smooth(ref[f"{axis}_int"], 1000, sigma_ms=20), ref[f"{axis}_gauss"])

@pytest.mark.parametrize("fs", [1000, 500, 250])
def test_gaussian_smooth_rate_invariant(fs):
    """Same sigma in ms -> same attenuation of a 5 Hz sine at any sampling rate."""
    f, sigma_ms = 5, 20
    t = np.arange(0, 2, 1 / fs)
    x = np.sin(2 * np.pi * f * t)
    expected = np.exp(-(2 * np.pi * f * sigma_ms / 1000) ** 2 / 2) * x
    mid = slice(int(0.2 * fs), int(1.8 * fs))          # skip edges, where 'reflect' padding differs
    np.testing.assert_allclose(gaussian_smooth(x, fs, sigma_ms)[mid], expected[mid], atol=1e-3)

def test_gaussian_smooth_nan_raises():
    with pytest.raises(ValueError):
        gaussian_smooth(np.array([1.0, np.nan, 2.0]), 1000)


def _samples(fs, duration, signal):
    """Build an (N, 4) array [timestamp_ms, x, y, pupil] with the same signal in all three columns."""
    t = np.arange(0, duration, 1 / fs)
    x = signal(t)
    return np.column_stack((1e6 + t * 1000, x, x, x)), t


def test_downsample_timestamps_evenly_spaced():
    s, _ = _samples(1000, 10, lambda t: np.sin(2 * np.pi * t))
    d = downsample(s, 1000, 100)
    assert d.shape == (1000, 4)
    assert d[0, 0] == s[0, 0]
    np.testing.assert_allclose(np.diff(d[:, 0]), 10.0)          # 100 Hz -> 10 ms


def test_downsample_constant_stays_constant():                  # includes the edges
    s, _ = _samples(1000, 10, lambda t: np.full_like(t, 3.7))
    np.testing.assert_allclose(downsample(s, 1000, 100)[:, 1:], 3.7)


def test_downsample_keeps_slow_sine():
    s, _ = _samples(1000, 10, lambda t: np.sin(2 * np.pi * t))
    d = downsample(s, 1000, 100)
    t_new = (d[:, 0] - 1e6) / 1000
    np.testing.assert_allclose(d[50:-50, 1], np.sin(2 * np.pi * t_new[50:-50]), atol=1e-3)


def test_downsample_removes_aliasing():
    """45 Hz is above the new Nyquist (25 Hz): it must be filtered out, not folded to 5 Hz."""
    s, _ = _samples(1000, 10, lambda t: np.sin(2 * np.pi * 45 * t))
    d = downsample(s, 1000, 50)
    assert np.max(np.abs(d[20:-20, 1])) < 0.01


def test_downsample_non_integer_ratio():                          # e.g. 10 samples per 1.2 s TR
    s, _ = _samples(1000, 12, lambda t: np.sin(2 * np.pi * 0.5 * t))
    d = downsample(s, 1000, 1000 / 120)
    t_new = (d[:, 0] - 1e6) / 1000
    assert d.shape[0] == 100
    np.testing.assert_allclose(d[5:-5, 1], np.sin(2 * np.pi * 0.5 * t_new[5:-5]), atol=1e-2)


def test_downsample_nan_raises():
    s, _ = _samples(1000, 1, lambda t: np.sin(2 * np.pi * t))
    s[100, 1] = np.nan
    with pytest.raises(ValueError):
        downsample(s, 1000, 100)


def test_downsample_upsampling_raises():
    s, _ = _samples(1000, 1, lambda t: np.sin(2 * np.pi * t))
    with pytest.raises(ValueError):
        downsample(s, 1000, 2000)


rng = np.random.default_rng(0)
N = 10_000
idx = np.arange(N)


# ---- regression: same result as the old drift_correction() ----

def test_drift_linear_matches_old_scipy_detrend():
    x = rng.normal(size=N) + 0.001 * idx + 2.0
    np.testing.assert_allclose(drift_correct(x, "linear"), detrend(x, type="linear"), atol=1e-10)


def test_drift_median_matches_old():
    x = rng.normal(size=N) + 2.0
    mask = rng.random(N) > 0.5
    np.testing.assert_allclose(drift_correct(x, "median", mask), x - np.median(x[mask]))


# ---- known answers ----

def test_drift_linear_removes_known_drift():
    x = rng.normal(scale=0.1, size=N) + 0.0005 * idx + 1.5     # 5 dva drift over the run + offset
    out = drift_correct(x, "linear")
    assert abs(np.polyfit(idx, out, 1)[0]) < 1e-6              # slope gone
    assert abs(np.mean(out)) < 1e-10                            # centred on reference 0


def test_drift_fit_uses_only_masked_samples():
    """Eye-movement periods (mask False) must not bias the drift estimate."""
    x = 0.0005 * idx.astype(float)                              # pure drift
    fix = (idx // 1000) % 2 == 0                                # alternate 1 s fixation / 1 s movement
    x[~fix] += 8.0                                              # gaze far away during movement blocks
    out = drift_correct(x, "linear", mask=fix)
    np.testing.assert_allclose(out[fix], 0.0, atol=1e-8)
    np.testing.assert_allclose(out[~fix], 8.0, atol=1e-8)


def test_drift_reference_position():
    x = rng.normal(scale=0.1, size=N) + 0.0005 * idx
    out = drift_correct(x, "linear", reference=-8.24)           # e.g. fixPos target 4
    assert abs(np.mean(out) - (-8.24)) < 1e-10


def test_drift_nans_ignored_and_kept():
    x = 0.0005 * idx + 1.0
    x[100:200] = np.nan
    out = drift_correct(x, "linear")
    assert np.isnan(out[100:200]).all()
    np.testing.assert_allclose(out[~np.isnan(out)], 0.0, atol=1e-8)


def test_drift_does_not_modify_input():
    x = rng.normal(size=N)
    x_copy = x.copy()
    drift_correct(x, "linear")
    np.testing.assert_array_equal(x, x_copy)


def test_drift_errors():
    x = rng.normal(size=N)
    with pytest.raises(ValueError):
        drift_correct(x, "nope")
    with pytest.raises(ValueError):
        drift_correct(x, "linear", mask=np.ones(N - 1, dtype=bool))
    with pytest.raises(ValueError):
        drift_correct(x, "linear", mask=np.zeros(N, dtype=bool))


def test_make_mask():
    t = np.arange(0, 10, 1.0)
    mask = make_mask(t, onsets=[2, 7], durations=[3, 1])
    np.testing.assert_array_equal(np.where(mask)[0], [2, 3, 4, 7])