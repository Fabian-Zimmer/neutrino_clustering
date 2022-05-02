"""
Some helper functions .

Contains the argument parser and default parsing results.

See the README for available argument parsers.
"""

import argparse as ap
import glob
from typing import Optional

snapshot_filename: Optional[str]

parser = ap.ArgumentParser(
    description="""General argument parser for SIDMPlots scripts."""
)

parser.add_argument(
    "-d",
    "--directory",
    help="Directory containing snapshots. Required.",
    type=str,
    required=True,
)

parser.add_argument(
    "-s",
    "--snapshot",
    help="Snapshot number to visualise. If not present, the latest snapshot is used.",
    type=int,
    required=True,
    default=None,
)

parser.add_argument(
    "-n",
    "--name",
    help="Simulation name. If not present, the name 'Simulation' is used.",
    type=str,
    required=True,
    default=None,
)

parser.add_argument(
    "-o",
    "--output",
    help="Output directory for the figures. If not present, the same directory as the snapshots are in is used.",
    required=False,
    default=None,
)

args = parser.parse_args()

# Now provide postprocessing to those arguments.

if args.snapshot is None:
    snapshot_filename = sorted(
        glob.glob(f"{args.directory}/eagle_*.hdf5"),
        key=lambda x: int(x[-9:-5]),
    )[-1]
    args.number = int(snapshot_filename[-9:-5])

if args.output is None:
    args.output = args.directory

if args.name is None:
    args.name = "Simulation"
