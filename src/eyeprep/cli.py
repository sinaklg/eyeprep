import argparse
from pathlib import Path
from .pipeline import run_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eyeprep",
        description="Eye-tracking preprocessing and reporting for BIDS datasets",
    )

    parser.add_argument("bids_path", type=Path, help="Path to BIDS dataset root")
    parser.add_argument("subject", help="Subject ID (e.g., 01 or sub-01)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML report path (default: BIDS derivatives folder)",
    )
    parser.add_argument(
        "--skip-bids-validation",
        action="store_true",
        help="Skip BIDS dataset validation",
    )

    args = parser.parse_args()

    subject = args.subject.replace("sub-", "")

    if args.output is None:
        output = (
            args.bids_path
            / "derivatives"
            / "eyeprep"
            / f"sub-{subject}"
            / "figures"
            / f"sub-{subject}_summary.html"
        )
    else:
        output = args.output

    output.parent.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        bids_path=args.bids_path,
        subject=subject,
        output_file=output,
        skip_bids_validation=args.skip_bids_validation
    )