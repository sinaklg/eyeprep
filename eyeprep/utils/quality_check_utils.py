
def compute_residuals(eye_data_df, eye_data_df_smoothed):
    import numpy as np
    import pandas as pd
    """
    Compute the residual standard deviation of gaze x and y after smoothing (residuals = raw - smoothed). 

    Args:
        eye_data_df (pd.DataFrame): Raw eye-tracking data with columns 'x' and 'y'.
        eye_data_df_smoothed (pd.DataFrame): Smoothed eye-tracking data with columns 'x' and 'y'.

    Returns:
        dict: Standard deviations of residuals for x and y coordinates.
            - 'x_std': standard deviation of x residuals
            - 'y_std': standard deviation of y residuals
    """

    # Compute residuals
    residuals = pd.DataFrame({
        'x': eye_data_df['x'] - eye_data_df_smoothed['x'],
        'y': eye_data_df['y'] - eye_data_df_smoothed['y']
    })

    # Return standard deviation of residuals
    return {
        'x_std': np.nanstd(residuals['x']),
        'y_std': np.nanstd(residuals['y'])
    }
