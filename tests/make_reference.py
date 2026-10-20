import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent / "eyeprep" / "utils"))     
from eyetrack_utils import blinkrm_pupil_off, blinkrm_pupil_off_smooth, convert_to_dva, interpol_nans



df = pd.read_csv(HERE / "data/sub-01/ses-01/func/sub-01_ses-01_task-SacLoc_run-01_recording-eye1_physio.tsv.gz",
                 sep="\t", header=None, na_values="n/a")
raw = df.to_numpy(dtype=float)[:200_000]           # a few minutes of data is enough

blinks = blinkrm_pupil_off(raw.copy(), 1000)
blinks_smooth = blinkrm_pupil_off_smooth(raw.copy(), 1000)
dva    = convert_to_dva(blinks.copy(), [960, 540], 52.0)
x_int  = interpol_nans(dva[:, 1].copy())

np.savez(HERE / "reference/sacloc_run01.npz", raw=raw, blinks=blinks, blinks_smooth = blinks_smooth, dva=dva, x_int=x_int)