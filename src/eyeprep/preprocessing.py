# preprocessing script 
# here all functions (will be called by pipeline )

import numpy as np


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