#!/usr/bin/env python3
"""
Query Heat Pump Electricity Rate from EnergyPlus SQL output (eplusout.sql).

Reads the variable "Heat Pump Electricity Rate" for key "ERGA08EAV3A 1"
and prints basic statistics (mean, median, min, max, std, count) to the terminal.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Heat Pump Electricity Rate from eplusout.sql"
    )
    parser.add_argument(
        "sql_file",
        nargs="?",
        default="Eplus-energyplus/eplusout.sql",
        help="Path to eplusout.sql (default: Eplus-energyplus/eplusout.sql)",
    )
    parser.add_argument(
        "--variable",
        default="Heat Pump Electricity Rate",
        help="Variable name (default: Heat Pump Electricity Rate)",
    )
    parser.add_argument(
        "--key",
        default="ERGA08EAV3A 1",
        help="KeyValue in the SQL dictionary (default: ERGA08EAV3A 1)",
    )
    args = parser.parse_args()

    path = Path(args.sql_file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # Get ReportVariableDataDictionaryIndex for this variable and key
    cur.execute(
        """
        SELECT ReportVariableDataDictionaryIndex
        FROM ReportVariableDataDictionary
        WHERE VariableName = ? AND KeyValue = ?
        """,
        (args.variable, args.key),
    )
    row = cur.fetchone()
    if not row:
        cur.execute(
            "SELECT KeyValue, VariableName FROM ReportVariableDataDictionary WHERE VariableName LIKE ? LIMIT 20",
            (f"%{args.variable}%",),
        )
        suggestions = cur.fetchall()
        print(f"Error: no record for variable '{args.variable}' and key '{args.key}'.", file=sys.stderr)
        if suggestions:
            print("Available options (KeyValue, VariableName):", file=sys.stderr)
            for kv, vn in suggestions:
                print(f"  {kv!r}, {vn!r}", file=sys.stderr)
        conn.close()
        sys.exit(1)

    dict_index = row[0]

    # Get all values for this variable
    cur.execute(
        """
        SELECT VariableValue
        FROM ReportVariableData
        WHERE ReportVariableDataDictionaryIndex = ?
        ORDER BY TimeIndex
        """,
        (dict_index,),
    )
    values = [r[0] for r in cur.fetchall()]
    conn.close()

    if not values:
        print("No data points found.", file=sys.stderr)
        sys.exit(1)

    n = len(values)
    if HAS_NUMPY:
        arr = np.array(values, dtype=float)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        std_val = float(np.std(arr))
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
    else:
        sorted_vals = sorted(values)
        mean_val = sum(values) / n
        min_val = min(values)
        max_val = max(values)
        variance = sum((x - mean_val) ** 2 for x in values) / n
        std_val = variance ** 0.5
        median_val = sorted_vals[n // 2] if n % 2 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        q25 = sorted_vals[int(0.25 * n)] if n else 0
        q75 = sorted_vals[int(0.75 * n)] if n else 0

    print(f"Variable: {args.variable}")
    print(f"Key:      {args.key}")
    print(f"File:     {path}")
    print()
    print(f"  Count:   {n}")
    print(f"  Mean:    {mean_val}")
    print(f"  Median:  {median_val}")
    print(f"  Std:     {std_val}")
    print(f"  Min:     {min_val}")
    print(f"  Max:     {max_val}")
    if HAS_NUMPY:
        print(f"  Q25:     {q25}")
        print(f"  Q75:     {q75}")


if __name__ == "__main__":
    main()
