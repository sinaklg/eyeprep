# preprocessing script 
# here all functions (will be called by pipeline )

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from fractions import Fraction
from scipy.signal import resample_poly

def _blink_mask(pupil, sampling_rate, addms2blink):
    """Boolean mask of samples with pupil == 0, extended by ± addms2blink."""
    ext = int(sampling_rate / 1000 * addms2blink)
    mask = np.zeros(len(pupil), dtype=bool)
    for idx in np.where(pupil == 0)[0]:
        mask[max(0, idx - ext):min(len(pupil), idx + ext + 1)] = True
    return mask


def remove_blinks(samples, method="pupil_off", sampling_rate=1000,
                  addms2blink=None, smoothing_duration=200):
    """
    Set blink samples to NaN. Returns a copy.

    method : 'pupil_off'        pupil == 0, extended by ± addms2blink (default 150 ms)
             'pupil_off_smooth' pupil == 0, extended by ± addms2blink (default 50 ms),
                                then widened by a smoothing_duration (ms) box kernel
    """
    if method == "pupil_off":
        mask = _blink_mask(samples[:, 3], sampling_rate, addms2blink or 150)
    elif method == "pupil_off_smooth":
        mask = _blink_mask(samples[:, 3], sampling_rate, addms2blink or 50)
        n = int(sampling_rate / 1000 * smoothing_duration)
        mask = np.convolve(mask, np.ones(n) / n, mode="same") > 0
    else:
        raise ValueError(f"Unknown blink removal method: {method!r}")
    cleaned = samples.copy()
    cleaned[mask, 1:] = np.nan
    return cleaned

def to_dva(samples, center, ppd):
    """Pixels -> dva, centred on screen, y up. Returns a copy."""
    out = samples.copy()
    out[:, 1] = (out[:, 1] - center[0]) / ppd
    out[:, 2] = -(out[:, 2] - center[1]) / ppd
    return out


def interpolate(signal):
    """Linear interpolation over NaNs, edges filled with nearest valid value. Returns a copy."""
    signal = signal.copy()
    nan = np.isnan(signal)
    return np.interp(np.arange(len(signal)), np.where(~nan)[0], signal[~nan])


def normalize_pupil(pupil, feature_range=(-1, 1)):
    """Min-max scale pupil size to feature_range. Returns a copy."""
    lo, hi = feature_range
    pmin, pmax = np.min(pupil), np.max(pupil)
    if pmax == pmin:
        return np.full_like(pupil, (lo + hi) / 2, dtype=float)
    return (pupil - pmin) / (pmax - pmin) * (hi - lo) + lo


def moving_average(signal, sampling_rate, window_ms):
    """Centred moving average. Edges (window/2) are NaN, as in the original eyeprep."""
    window = int(sampling_rate * window_ms / 1000)
    return pd.Series(signal).rolling(window=window, center=True).mean().to_numpy()


def gaussian_smooth(signal, sampling_rate, sigma_ms=20, truncate=4.0, mode="reflect"):
    """
    Gaussian smoothing of a 1D signal.

    sigma_ms : standard deviation of the kernel in ms (converted to samples with sampling_rate)
    truncate : kernel extends to truncate * sigma on each side (default ±80 ms)
    mode     : edge handling, passed to scipy ('reflect' -> no NaN edges)

    Raises ValueError if the signal contains NaNs: run interpolation first.
    """
    signal = np.asarray(signal, dtype=float)
    if np.isnan(signal).any():
        raise ValueError("gaussian_smooth: signal contains NaNs, interpolate before smoothing")
    return gaussian_filter1d(signal, sigma=sampling_rate * sigma_ms / 1000,
                             truncate=truncate, mode=mode)


def downsample(samples, sampling_rate, target_rate, padtype="line"):
    """
    Downsample [timestamp_ms, x, y, pupil] data with an anti-aliasing filter.

    Timestamps are rebuilt on an exact grid (t0 + k / target_rate).
    Signals use scipy.signal.resample_poly (polyphase FIR filter), which handles
    non-integer ratios such as 1000 Hz -> 1000/120 Hz (10 samples per 1.2 s TR).

    padtype : edge padding for the filter. 'line' avoids the edge droop of
              scipy's default zero padding.

    Raises ValueError on NaNs (interpolate first) or if target_rate >= sampling_rate.
    """
    samples = np.asarray(samples, dtype=float)
    if target_rate >= sampling_rate:
        raise ValueError("target_rate must be lower than sampling_rate")
    if np.isnan(samples[:, 1:]).any():
        raise ValueError("downsample: signal contains NaNs, interpolate before downsampling")

    ratio = Fraction(target_rate / sampling_rate).limit_denominator(1000)
    signals = resample_poly(samples[:, 1:], ratio.numerator, ratio.denominator,
                            axis=0, padtype=padtype)
    t = samples[0, 0] + np.arange(signals.shape[0]) * (1000 / target_rate)
    return np.column_stack((t, signals))     # (N, 4): fixes the old transpose bug

def make_mask(timestamps, onsets, durations):
    """Boolean mask, True where timestamps fall inside any [onset, onset + duration) window (same units)."""
    timestamps = np.asarray(timestamps, dtype=float)
    mask = np.zeros(len(timestamps), dtype=bool)
    for onset, dur in zip(onsets, durations):
        mask |= (timestamps >= onset) & (timestamps < onset + dur)
    return mask


def drift_correct(signal, method="linear", mask=None, reference=0.0):
    """
    Remove slow drift from a 1D gaze signal. Returns a copy.

    method    : 'linear' - fit a line on the masked samples, subtract it from the whole signal
                'median' - subtract the median of the masked samples (offset only)
    mask      : boolean array, True for samples where gaze is assumed to be at `reference`
                (e.g. fixation periods). None = all samples.
    reference : position (dva) the masked samples are assumed to sit at, e.g. the fixation
                target. 0 = screen centre.

    NaNs are ignored for the fit and kept in the output.
    """
    signal = np.asarray(signal, dtype=float)
    mask = np.ones(len(signal), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != signal.shape:
        raise ValueError("mask must have the same length as signal")
    use = mask & ~np.isnan(signal)
    if use.sum() < 2:
        raise ValueError("drift_correct: fewer than 2 valid samples in mask")

    if method == "linear":
        idx = np.arange(len(signal))
        trend = np.polyval(np.polyfit(idx[use], signal[use], deg=1), idx)
    elif method == "median":
        trend = np.median(signal[use])
    else:
        raise ValueError(f"Unknown drift correction method: {method!r}")
    return signal - trend + reference