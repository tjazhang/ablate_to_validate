#!/usr/bin/env python3
# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: data/convert_depth_point_to_parquet.py

"""Convert depth_point_hf_viewer.json to a Hugging Face-ready Parquet.

This script reads a JSON array file containing records with an `image` path,
loads each image as bytes, constructs an Arrow table where the `image` column
is a struct with `bytes` and `path`, annotates the schema so the Hub renders
the column as an Image, and writes a Parquet file using content-defined
chunking.

Usage:
  python convert_depth_point_to_parquet.py \
    --input /path/to/depth_point_hf_viewer.json \
    --output /path/to/depth_point_hf_viewer.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import json as _json


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSON to HF-ready Parquet with Image bytes.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to depth_point_hf_viewer.json",
    )
    parser.add_argument(
        "--output",
        required=False,
        type=Path,
        help="Path to output Parquet. Defaults to input path with .parquet extension.",
    )
    return parser.parse_args()


def load_records(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_table(records: List[Dict[str, Any]], base_dir: Path) -> pa.Table:
    if not records:
        return pa.table({})

    first_record_keys = list(records[0].keys())

    column_name_to_values: Dict[str, List[Any]] = {}
    for key in first_record_keys:
        if key == "image":
            continue
        column_name_to_values[key] = []

    image_struct_values: List[Dict[str, Any]] = []

    for record in records:
        for key, values in column_name_to_values.items():
            values.append(record.get(key))

        image_rel_path = record.get("image")
        if image_rel_path is None:
            image_struct_values.append({"bytes": None, "path": None})
            continue

        image_abs_path = (base_dir / image_rel_path).resolve()
        image_bytes: bytes = image_abs_path.read_bytes()
        image_struct_values.append({"bytes": image_bytes, "path": str(image_rel_path)})

    base_table = pa.table({name: pa.array(values) for name, values in column_name_to_values.items()})
    image_type = pa.struct([pa.field("bytes", pa.binary()),
                           pa.field("path", pa.string())])
    image_array = pa.array(image_struct_values, type=image_type)
    table = base_table.append_column("image", image_array)

    # Annotate schema so HF Hub renders the column as an Image.
    features = {"image": {"_type": "Image"}}
    hf_metadata = _json.dumps({"dataset_info": {"features": features}}).encode("utf-8")
    schema_metadata = {b"huggingface": hf_metadata}
    table = table.replace_schema_metadata(schema_metadata)
    return table


def main() -> None:
    args = parse_arguments()
    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_suffix(".parquet")

    records = load_records(input_path)
    base_dir = input_path.parent

    table = build_table(records, base_dir)

    pq.write_table(table, str(output_path), use_content_defined_chunking=True, row_group_size=100)



if __name__ == "__main__":
    main()

