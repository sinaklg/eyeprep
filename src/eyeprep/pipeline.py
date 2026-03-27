from pathlib import Path
from datetime import datetime
import sys
import getpass
from collections import defaultdict

from bids import BIDSLayout
from . import __version__



# Data collection
def collect_tasks(layout: BIDSLayout, subject: str) -> dict:
    """
    Collect eye-tracking files and group by task.

    Returns:
    {
        task_name: [
            {"run": int, "eye": str, "file": str}
        ]
    }
    """
    files = layout.get(
        subject=subject,
        extension="tsv.gz",
        return_type="filename",
    )

    tasks = defaultdict(list)

    for f in files:
        entities = layout.parse_file_entities(f)

        task = entities.get("task", "unknown")
        run = entities.get("run", 1)

        # ---- Eye detection (heuristic for now) ----
        fname = Path(f).name.lower()

        if "eye1" in fname or "left" in fname:
            eye = "left"
        elif "eye2" in fname or "right" in fname:
            eye = "right"
        else:
            eye = "unknown"

        tasks[task].append(
            {
                "run": run,
                "eye": eye,
                "file": Path(f).name,
            }
        )

    return dict(tasks)


# Report
def create_summary_report(subject: str, tasks: dict, metadata: dict) -> str:
    html = f"""
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

        <h2>Tasks (eye-tracking)</h2>
        <ul>
    """

    if tasks:
        for task, runs in tasks.items():
            html += f"<li>Task: {task} ({runs} runs)</li>\n"
    else:
        html += "<li>No eye-tracking files found</li>\n"

    html += f"""
        </ul>

        <div class="metadata">
            <p>BIDS validation: {metadata['bids_validation']}</p>
            <p>Eyeprep version: {metadata['version']}</p>
            <p>Command used: {metadata['command']}</p>
            <p>Date run: {metadata['date']}</p>
            <p>User: {metadata['user']}</p>
        </div>
    </body>
    </html>
    """

    return html



# Main pipeline
def run_pipeline(
    bids_path: Path,
    subject: str,
    output_file: Path,
    skip_bids_validation: bool = False,
):
    if not bids_path.exists():
        raise FileNotFoundError(f"BIDS path does not exist: {bids_path}")

    # --- Metadata ---
    metadata = {
        "command": " ".join(sys.argv),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": getpass.getuser(),
        "version": __version__,
        "bids_validation": "Skipped" if skip_bids_validation else "Performed",
    }

    # --- BIDS layout (inline, no abstraction) ---
    print("BIDS validation:", "SKIPPED" if skip_bids_validation else "ENABLED")

    layout = BIDSLayout(
        str(bids_path),
        validate=not skip_bids_validation,
    )

    # --- Collect data ---
    tasks = collect_tasks(layout, subject)
    print("Detected tasks:", tasks)

    # --- Generate report ---
    html_content = create_summary_report(subject, tasks, metadata)

    # --- Write output ---
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_content)

    print(f"Report written to {output_file}")