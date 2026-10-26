"""Loading and saving eye-tracking data."""
import json
import re
from pathlib import Path

import pandas as pd

DEEPMREYE_PATTERN = re.compile(
    r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_task-(?P<task>.+?)_run-0*(?P<run>\d+)"
    r"_pred_(?P<resolution>median|subtr)\.tsv\.gz$"
)

def convert_deepmreye(tsv_path):
    """
    Read an int_deepmreye prediction file + sidecar into eyeprep's internal format.

    Returns
    -------
    df : DataFrame with columns timestamp (ms), x (deg), y (deg)
    meta : dict with SamplingFrequency, units, source, and BIDS entities parsed from the filename
    """
    tsv_path = Path(tsv_path)
    match = DEEPMREYE_PATTERN.search(tsv_path.name)
    if match is None:
        raise ValueError(f"{tsv_path.name} does not look like an int_deepmreye prediction file")
    if match["session"].startswith("ses-"):
        raise ValueError(f"{tsv_path.name}: doubled session prefix (ses-ses-), rerun int_deepmreye with the fixed filenames")
    json_path = Path(str(tsv_path).replace(".tsv.gz", ".json"))
    if not json_path.exists():
        raise FileNotFoundError(f"Missing sidecar {json_path.name}: run write_deepmreye_sidecar first")
    sidecar = json.loads(json_path.read_text())

    raw = pd.read_csv(tsv_path, sep="\t")
    expected = sidecar["Columns"]
    if list(raw.columns) != expected:
        raise ValueError(f"{tsv_path.name}: columns {list(raw.columns)} do not match sidecar {expected}")

    to_ms = {"s": 1000.0, "ms": 1.0}[sidecar["timestamp"]["Units"]]
    df = pd.DataFrame({
        "timestamp": raw["timestamp"].to_numpy() * to_ms,
        "x": raw["X"].to_numpy(),
        "y": raw["Y"].to_numpy(),
    })
    meta = {
        "source": "deepmreye",
        "SamplingFrequency": sidecar["SamplingFrequency"],
        "units": sidecar["X"]["Units"],
        "has_pupil": False,
        **match.groupdict(),
        "sidecar": sidecar,
    }
    meta["run"] = int(meta["run"])
    return df, meta