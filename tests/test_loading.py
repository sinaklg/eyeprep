import numpy as np
import pandas as pd
import pytest
import json
from eyeprep.loading import convert_deepmreye
# write_deepmreye_sidecar: copy it into tests/ or import from int_deepmreye; it's only used to build fake data

# ---- helpers: build fake int_deepmreye output -------------------------------------------

def write_deepmreye_sidecar(tsv_path, tr, n_subtr):
    """Same sidecar as int_deepmreye writes (copy kept here so tests don't depend on int_deepmreye)."""
    sidecar = {
        "PhysioType": "eyetrack",
        "SamplingFrequency": n_subtr / tr,
        "StartTime": 0.0,
        "Columns": ["timestamp", "X", "Y"],
        "RecordedEye": "cyclopean",
        "SampleCoordinateSystem": "gaze-on-screen",
        "EnvironmentCoordinates": "center",
        "RepetitionTime": tr,
        "SubTRs": n_subtr,
        "timestamp": {"Description": "Time from the onset of the first volume", "Units": "s"},
        "X": {"Description": "Predicted gaze position x (positive = right)", "Units": "deg"},
        "Y": {"Description": "Predicted gaze position y (positive = up)", "Units": "deg"},
        "GeneratedBy": [{"Name": "int_deepmreye"}],
    }
    tsv_path.with_name(tsv_path.name.replace(".tsv.gz", ".json")).write_text(json.dumps(sidecar, indent=4))

def _fake_pred(tmp_path, name, n, tr=1.2, n_subtr=1):
    """Write a file exactly like int_deepmreye does (header timestamp/X/Y, gzip), plus its sidecar."""
    path = tmp_path / name
    df = pd.DataFrame({"timestamp": np.arange(n) * (tr / n_subtr),
                       "X": np.linspace(-5, 5, n), "Y": np.zeros(n)})
    df.to_csv(path, sep="\t", index=False, compression="gzip")
    write_deepmreye_sidecar(path, tr, n_subtr)
    return path


def test_convert_median(tmp_path):
    p = _fake_pred(tmp_path, "sub-01_ses-02_task-DeepMReyeCalib_run-01_pred_median.tsv.gz", 154)
    df, meta = convert_deepmreye(p)
    assert list(df.columns) == ["timestamp", "x", "y"]
    np.testing.assert_allclose(np.diff(df["timestamp"]), 1200.0)        # seconds -> ms
    assert meta["SamplingFrequency"] == pytest.approx(1 / 1.2)
    assert (meta["subject"], meta["session"], meta["task"], meta["run"]) == ("01", "02", "DeepMReyeCalib", 1)
    assert meta["resolution"] == "median" and meta["units"] == "deg" and not meta["has_pupil"]


def test_convert_subtr(tmp_path):
    p = _fake_pred(tmp_path, "sub-03_ses-02_task-pRF_run-10_pred_subtr.tsv.gz", 1540, n_subtr=10)
    df, meta = convert_deepmreye(p)
    np.testing.assert_allclose(np.diff(df["timestamp"]), 120.0)
    assert meta["SamplingFrequency"] == pytest.approx(10 / 1.2)
    assert meta["run"] == 10 and meta["resolution"] == "subtr"


def test_convert_double_ses_prefix_raises(tmp_path):                # old files from before the fix
    p = _fake_pred(tmp_path, "sub-01_ses-ses-02_task-x_run-01_pred_median.tsv.gz", 10)
    with pytest.raises(ValueError, match="doubled session"):
        convert_deepmreye(p)


def test_convert_missing_sidecar(tmp_path):
    p = _fake_pred(tmp_path, "sub-01_ses-02_task-x_run-01_pred_median.tsv.gz", 10)
    p.with_name(p.name.replace(".tsv.gz", ".json")).unlink()
    with pytest.raises(FileNotFoundError):
        convert_deepmreye(p)


def test_convert_wrong_name(tmp_path):
    p = _fake_pred(tmp_path, "something_else.tsv.gz", 10)
    with pytest.raises(ValueError):
        convert_deepmreye(p)