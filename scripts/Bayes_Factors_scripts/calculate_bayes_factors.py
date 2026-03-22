#!/usr/bin/env python3
import math
from pathlib import Path

# Input and output paths
INPUT_FILE = Path("/fred/oz002/users/mmiles/MPTA_GW/enterprise_newdata/out_pbilby_pp_8/J1713+0747/trial_sort_sgwb.list")
OUTPUT_FILE = Path("J1713_meerkatBF.txt")

# Allowed / excluded tags
ALLOWED_TAGS = {"PM", "EFAC", "EQUAD", "ECORR", "RN", "DM", "CHROM", "SW", "SGWB", "SWDET", "CHROMANNUAL", "CHROMCIDX", "SMBHB"}
EXCLUDE_SUBSTRINGS = {"CHROMBUMP"}#"SWDET", "CHROMANNUAL", "CHROMBUMP", "CHROMCIDX", "SMBHB"}

def parse_line(line):
    parts = line.strip().split()
    if len(parts) != 2:
        return None, None
    name, val = parts
    try:
        logZ = float(val)
    except ValueError:
        return None, None
    return name, logZ

def is_valid_model(name):
    if any(x in name for x in EXCLUDE_SUBSTRINGS):
        return False
    tokens = name.split("_")
    if tokens[-1] != "SGWB":
        return False
    comps = tokens[1:]
    return set(comps).issubset(ALLOWED_TAGS)

def main():
    evidences = {}
    with INPUT_FILE.open() as f:
        for line in f:
            name, logZ = parse_line(line)
            if name and is_valid_model(name):
                evidences[name] = logZ

    if not evidences:
        raise RuntimeError("No valid models found in file!")

    # Reference = highest logZ model
    ref_name = max(evidences, key=lambda k: evidences[k])
    ref_logZ = evidences[ref_name]

    results = []
    for name, logZ in evidences.items():
        dlogZ = logZ - ref_logZ
        try:
            B = math.exp(dlogZ)
        except OverflowError:
            B = float("inf") if dlogZ > 0 else 0.0
        results.append((name, logZ, dlogZ, B))

    results.sort(key=lambda x: x[2], reverse=True)

    lines = []
    lines.append(f"Reference model: {ref_name}")
    lines.append(f"Reference logZ: {ref_logZ:.6f}")
    lines.append("---------------------------------------------------------------")
    lines.append(f"{'MODEL':60s}  {'logZ':>12s}  {'ΔlogZ':>10s}  {'BayesFactor':>12s}")
    lines.append("---------------------------------------------------------------")
    for name, logZ, dlogZ, B in results:
        B_str = f"{B:.6e}" if math.isfinite(B) else ("inf" if dlogZ > 0 else "0")
        lines.append(f"{name:60s}  {logZ:12.6f}  {dlogZ:10.6f}  {B_str:>12s}")

    output_text = "\n".join(lines)

    # Print and save
    print(output_text)
    OUTPUT_FILE.write_text(output_text + "\n")
    print(f"\nSaved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
