from pathlib import Path
from datetime import datetime
import sys
import getpass
from bids import BIDSLayout

# import the version
from . import __version__

from bids_validator import BIDSValidator

def run_pipeline(bids_path: Path, subject: str, output_file: Path, skip_bids_validation: bool = False):
    """
    Generate a simple HTML summary report for a subject.
    Currently skeleton: only placeholder counts and tasks.
    """
    if not bids_path.exists():
        raise FileNotFoundError(f"BIDS path does not exist: {bids_path}")
    
    # --- Metadata ---
    command_used = " ".join(sys.argv)
    date_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = getpass.getuser()

    # --- BIDS validation ---
    print("BIDS validation:", "SKIPPED" if skip_bids_validation else "ENABLED")

    layout = BIDSLayout(
        str(bids_path),
        validate=not skip_bids_validation,
    )

    bids_validation_status = "Skipped" if skip_bids_validation else "Performed"

    # --- Placeholders for now ---
    #TODO use pybids to find them 
    tasks = {
        "SacLoc": 2,
        "SacVELoc": 2,
        "pMF": 5,
    }
    # --- Generate HTML ---
    html_content = f"""
    <html>
    <head>
        <title>Eyeprep report for sub-{subject}</title>
        <style>
            body {{ font-family: sans-serif; }}
            h1 {{ color: #2c3e50; }}
            ul {{ list-style-type: none; padding-left: 0; }}
            li {{ margin-bottom: 4px; }}
            .metadata {{ font-size: 0.9em; color: #555; }}
        </style>
    </head>
    <body>
        <h1>Eyeprep summary for sub-{subject}</h1>

        <h2>Subject information</h2>
        <ul>
            <li>Subject ID: {subject}</li>
        </ul>

        <h2>Tasks (eye-tracking only)</h2>
        <ul>
    """
    for task_name, runs in tasks.items():
        html_content += f"            <li>Task: {task_name} ({runs} runs)</li>\n"

    html_content += f"""
        </ul>
        <div class="metadata">
            <p>BIDS validation: {bids_validation_status}</p>
            <p>Eyeprep version: {__version__}</p>
            <p>Command used: {command_used}</p>
            <p>Date run: {date_run}</p>
            <p>User: {user}</p>
        </div>
    </body>
    </html>
    """

    # Ensure parent folder exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write the report
    output_file.write_text(html_content)
    print(f"Report written to {output_file}")