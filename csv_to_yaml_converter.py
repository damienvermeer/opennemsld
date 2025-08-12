#!/usr/bin/env python3
"""
CSV to YAML Converter for Substation Data

This script converts digital.atlas.gov.au CSV data into nemsld YAML format
(locations and voltages only).
"""

import csv
import yaml
import os


def convert_csv_to_yaml(
    csv_file_path, output_yaml_filename, min_voltage_kv=None, max_voltage_kv=None
):
    """
    Convert digital.atlas.gov.au CSV data into nemsld YAML format (locations and voltages only).

    Args:
        csv_file_path: Full file path to the CSV file to read
        output_yaml_filename: Output YAML filename (saved in sld-data folder)
        min_voltage_kv: Optional minimum voltage in kV to filter substations
        max_voltage_kv: Optional maximum voltage in kV to filter substations
    """
    # Define output file path relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(base_dir, "sld-data", output_yaml_filename)

    # Initialize the output data structure
    output_data = {"substations": []}

    # Read the CSV data
    with open(csv_file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Convert voltage to float (handling potential empty or non-numeric values)
            try:
                voltage = float(row["voltagekv"])
            except (ValueError, TypeError):
                continue

            # Apply voltage filtering if specified
            if min_voltage_kv is not None and voltage <= min_voltage_kv:
                continue
            if max_voltage_kv is not None and voltage >= max_voltage_kv:
                continue
            else:
                substation = {
                    "name": row["name"],
                    "lat": float(
                        row["y_coordinate"]
                    ),  # Y coordinate corresponds to latitude
                    "long": float(
                        row["x_coordinate"]
                    ),  # X coordinate corresponds to longitude
                    "voltage_kv": int(voltage) if voltage.is_integer() else voltage,
                    "def": "?|?",
                    "buses": {1: ""},
                    "connections": {},
                }

                # Add the substation to our output data
                output_data["substations"].append(substation)

    # Write the YAML output
    with open(output_file, "w", encoding="utf-8") as yamlfile:
        yaml.dump(output_data, yamlfile, default_flow_style=False, sort_keys=False)

    print(f"Conversion complete! Output saved to {output_file}")
    print(f"Total substations processed: {len(output_data['substations'])}")


if __name__ == "__main__":
    # Example usage with default parameters
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "sld-data", "sadata.csv")
    convert_csv_to_yaml(input_file, "sa_subtransmission.yaml", min_voltage_kv=132)
