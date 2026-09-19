"""
District Data Cleaner for NCRB 2024 Tier 1 Datasets.

Extracts administrative district records, normalizes state and district identities,
removes summary/header artifacts, and prefixes columns with domain tags.
"""

import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import openpyxl
import pandas as pd

DOMAIN_PREFIX_MAP = {
    "1DistrictwiseIPCCrimes2024.xlsx": "ipc",
    "2DistrictwiseSLLCrimes2024.xlsx": "sll",
    "3DistrictwiseCrimeagainstWomen2024.xlsx": "women",
    "4DistrictwiseCrimeagainstChildren2024.xlsx": "child",
    "5DistrictwiseCrimeagainstSCs2024.xlsx": "sc",
    "6DistrictwiseCrimeagainstSTs2024.xlsx": "st",
    "7DistrictwiseIPCCrimebyJuveniles2024.xlsx": "juv_ipc",
    "9DistrictwiseCyberCrimes2024.xlsx": "cyber",
    "10DistrictwiseMissingPersons2024.xlsx": "missing",
}


def sanitize_column_name(text: str, prefix: str) -> str:
    """Normalize raw NCRB header strings into clean, readable snake_case column names."""
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)
    # Strip illegal characters, replace punctuation and spaces with underscores
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text).strip("_").lower()
    if not text:
        text = "unnamed"
    return f"{prefix}_{text}"


def normalize_entity_name(name: str) -> str:
    """Standardize state and district names for consistent cross-table joining."""
    if not name:
        return ""
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    # Remove leading numbering like '1. ' if present
    name = re.sub(r"^\d+[\.\)]\s*", "", name)
    return name.title()


def clean_single_district_file(
    file_path: Path, prefix: Optional[str] = None
) -> pd.DataFrame:
    """Parse a single NCRB district-level workbook into a clean, normalized DataFrame."""
    if prefix is None:
        prefix = DOMAIN_PREFIX_MAP.get(file_path.name, "feat")

    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb[wb.sheetnames[0]]
    all_rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not all_rows:
        return pd.DataFrame()

    num_cols = len(all_rows[1]) if len(all_rows) > 1 else len(all_rows[0])

    # Construct hierarchical column headers from rows 1 to 3
    col_names = []
    for c in range(num_cols):
        parts = []
        for r in range(1, 4):
            if r < len(all_rows) and c < len(all_rows[r]):
                val = all_rows[r][c]
                if val is not None:
                    val_str = str(val).strip().replace("\n", " ")
                    if val_str and val_str not in parts and not re.match(r"^\[?\d+\]?$", val_str):
                        parts.append(val_str)
        raw_header = " ".join(parts) if parts else f"col_{c}"
        col_names.append(raw_header)

    # Sanitize column names with domain prefix
    clean_cols = ["state", "district"]
    for c_idx in range(2, num_cols):
        clean_cols.append(sanitize_column_name(col_names[c_idx], prefix))

    clean_records = []
    current_state = "UNKNOWN"

    for r_idx in range(4, len(all_rows)):
        row = all_rows[r_idx]
        if not row or all(x is None for x in row):
            continue

        c0 = str(row[0]).strip() if row[0] is not None else ""
        c1 = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""

        # Detect State header
        full_text = f"{c0} {c1}".strip()
        if "State:" in full_text or "UT:" in full_text or (c0.startswith("State") and not c1):
            state_raw = (
                full_text.replace("State:", "")
                .replace("UT:", "")
                .replace("State / UT:", "")
                .strip()
            )
            if state_raw:
                current_state = normalize_entity_name(state_raw)
            continue

        # Skip summary rows and table footer notes
        lowered = f"{c0} {c1}".lower()
        if any(term in lowered for term in [
            "total", "source:", "all-india", "all india", "table", "as per data"
        ]):
            continue

        # Filter out column numbering index rows (e.g. '[1]', '[2]')
        district_raw = c1 if c1 else c0
        if not district_raw or re.match(r"^\[?\d+\]?$", district_raw):
            continue

        norm_state = normalize_entity_name(current_state)
        norm_district = normalize_entity_name(district_raw)

        # Entity reconciliation: Map 'All Districts' in Chandigarh to 'Chandigarh'
        if norm_state.upper() == "CHANDIGARH" and norm_district.upper() in ["ALL DISTRICTS", "TOTAL DISTRICTS"]:
            norm_district = "Chandigarh"

        record = {"state": norm_state, "district": norm_district}

        for c_idx in range(2, min(num_cols, len(row))):
            val = row[c_idx]
            col_key = clean_cols[c_idx]
            if val is None or str(val).strip() in ["-", "", "None", "NA", "N.A.", "*"]:
                record[col_key] = 0
            else:
                try:
                    record[col_key] = float(val) if "." in str(val) else int(val)
                except (ValueError, TypeError):
                    record[col_key] = 0

        clean_records.append(record)

    df = pd.DataFrame(clean_records)
    # Deduplicate in case of duplicate entries
    df = df.drop_duplicates(subset=["state", "district"])
    return df


def build_master_district_matrix(curated_dir: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load all 9 Tier 1 district files, clean each, and merge on state + district."""
    district_files = sorted(list(curated_dir.glob("*.xlsx")))
    cleaned_tables = {}

    for f in district_files:
        df_clean = clean_single_district_file(f)
        cleaned_tables[f.name] = df_clean

    # Use 1DistrictwiseIPCCrimes as master anchor
    anchor_name = "1DistrictwiseIPCCrimes2024.xlsx"
    master_df = cleaned_tables[anchor_name].copy()

    for name, df in cleaned_tables.items():
        if name == anchor_name:
            continue
        # Outer merge on state and district
        master_df = pd.merge(
            master_df,
            df,
            on=["state", "district"],
            how="outer",
            suffixes=("", f"_{name[:3]}"),
        )

    # Impute any missing numerical values with 0
    numeric_cols = master_df.select_dtypes(include=["number"]).columns
    master_df[numeric_cols] = master_df[numeric_cols].fillna(0)

    # Sort deterministically
    master_df = master_df.sort_values(by=["state", "district"]).reset_index(drop=True)
    return master_df, cleaned_tables
