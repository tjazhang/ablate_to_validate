"""Aurora-specific dataset resolution for Mull eval overrides.

Upstream Video-R1 eval scripts hard-code dataset mounts from one machine. The
Aurora overlay keeps those locations configurable so the ablation entrypoints
stay portable after the overlay is materialized into `external/mull/upstream`.
"""

import os


DEFAULT_BLINK_LOCATION = "BLINK-Benchmark/BLINK"
DEFAULT_SAT_LOCATION = "array/SAT-v2"
DEFAULT_SAT_SPLIT = "test"


def add_dataset_override_args(parser):
    """Register CLI flags for datasets that live outside the repo checkout."""
    parser.add_argument(
        "--blink-location",
        type=str,
        default=os.environ.get("MULL_BLINK_LOCATION", DEFAULT_BLINK_LOCATION),
        help=(
            "BLINK dataset source. Defaults to the Hugging Face dataset id "
            f"`{DEFAULT_BLINK_LOCATION}` or $MULL_BLINK_LOCATION."
        ),
    )
    parser.add_argument(
        "--sat-location",
        type=str,
        default=os.environ.get("MULL_SAT_LOCATION", DEFAULT_SAT_LOCATION),
        help=(
            "SAT dataset source. Defaults to the Hugging Face dataset id "
            f"`{DEFAULT_SAT_LOCATION}` or $MULL_SAT_LOCATION."
        ),
    )
    parser.add_argument(
        "--sat-split",
        type=str,
        default=os.environ.get("MULL_SAT_SPLIT", DEFAULT_SAT_SPLIT),
        help="SAT split to request. Defaults to `test` or $MULL_SAT_SPLIT.",
    )
    parser.add_argument(
        "--vsibench-location",
        type=str,
        default=os.environ.get("MULL_VSIBENCH_LOCATION"),
        help="Path to `eval_vsibench.json` or set $MULL_VSIBENCH_LOCATION.",
    )
    parser.add_argument(
        "--vsibench-root",
        type=str,
        default=os.environ.get("MULL_VSIBENCH_ROOT"),
        help="Directory with VSI-Bench videos or set $MULL_VSIBENCH_ROOT.",
    )
    parser.add_argument(
        "--mmvu-location",
        type=str,
        default=os.environ.get("MULL_MMVU_LOCATION"),
        help="Path to `eval_mmvu.json` or set $MULL_MMVU_LOCATION.",
    )
    parser.add_argument(
        "--mmvu-root",
        type=str,
        default=os.environ.get("MULL_MMVU_ROOT"),
        help="Directory with MMVU videos or set $MULL_MMVU_ROOT.",
    )


def _require(value, dataset_name, flag_name, env_name):
    if value:
        return value
    raise ValueError(
        f"Dataset `{dataset_name}` requires `{flag_name}` or `${env_name}`."
    )


def resolve_dataset_entries(dataset_names, args):
    """Build the dataset spec expected by Video-R1's EvalDataset loader."""
    entries = []
    for dataset_name in dataset_names:
        if dataset_name == "vsibench":
            entries.append(
                {
                    "dataset_name": "vsibench",
                    "vsibench_location": _require(
                        args.vsibench_location,
                        dataset_name,
                        "--vsibench-location",
                        "MULL_VSIBENCH_LOCATION",
                    ),
                    "vsibench_root": _require(
                        args.vsibench_root,
                        dataset_name,
                        "--vsibench-root",
                        "MULL_VSIBENCH_ROOT",
                    ),
                }
            )
        elif dataset_name == "blink":
            entries.append(
                {
                    "dataset_name": "blink",
                    "blink_location": args.blink_location,
                }
            )
        elif dataset_name == "sat":
            entries.append(
                {
                    "dataset_name": "sat",
                    "sat_location": args.sat_location,
                    "split": args.sat_split,
                }
            )
        elif dataset_name == "mmvu":
            entries.append(
                {
                    "dataset_name": "mmvu",
                    "mmvu_location": _require(
                        args.mmvu_location,
                        dataset_name,
                        "--mmvu-location",
                        "MULL_MMVU_LOCATION",
                    ),
                    "mmvu_root": _require(
                        args.mmvu_root,
                        dataset_name,
                        "--mmvu-root",
                        "MULL_MMVU_ROOT",
                    ),
                }
            )
        else:
            raise ValueError(
                f"Unsupported dataset `{dataset_name}`. "
                "Expected one of: vsibench, blink, sat, mmvu."
            )

    return entries
