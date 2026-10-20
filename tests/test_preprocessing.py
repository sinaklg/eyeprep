import numpy as np
import pytest
from eyeprep.preprocessing import remove_blinks, to_dva, interpolate   # your NEW functions
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