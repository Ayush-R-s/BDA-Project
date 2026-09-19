"""
Master Preprocessing Pipeline Runner for NCRB 2024 Crime Analytics Suite.

Executes end-to-end data cleaning, normalization, joining, and feature engineering
across all 17 curated datasets. Exports Parquet and CSV files to data/processed/.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.preprocessing.district_cleaner import (
    build_master_district_matrix,
    clean_single_district_file,
)
from src.preprocessing.metro_cleaner import (
    build_metro_master_trends,
    clean_metro_generic_table,
    parse_metro_juvenile_crime_demography,
)
from src.preprocessing.feature_engineering import (
    compute_district_features,
    compute_metro_features,
)


def run_pipeline():
    print("=" * 80)
    print("NCRB 2024 DATA PREPROCESSING PIPELINE")
    print("=" * 80)

    curated_dist_dir = PROJECT_ROOT / "data" / "curated" / "district"
    curated_metro_dir = PROJECT_ROOT / "data" / "curated" / "metropolitan"

    proc_dir = PROJECT_ROOT / "data" / "processed"
    proc_dist_dir = proc_dir / "district"
    proc_metro_dir = proc_dir / "metropolitan"

    proc_dist_dir.mkdir(parents=True, exist_ok=True)
    proc_metro_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. TIER 1: DISTRICT DATA PREPROCESSING
    # ==========================================
    print("\n[Phase 1/4] Processing Tier 1 District Datasets...")
    master_district_df, cleaned_district_tables = build_master_district_matrix(curated_dist_dir)

    for fname, df in cleaned_district_tables.items():
        base_name = Path(fname).stem
        parquet_path = proc_dist_dir / f"{base_name}.parquet"
        csv_path = proc_dist_dir / f"{base_name}.csv"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        print(f"  Exported: {base_name:<42} -> Shape: {df.shape}")

    # Export Master District Feature Matrix
    dist_master_parquet = proc_dir / "district_master_feature_matrix.parquet"
    dist_master_csv = proc_dir / "district_master_feature_matrix.csv"
    master_district_df.to_parquet(dist_master_parquet, index=False)
    master_district_df.to_csv(dist_master_csv, index=False)
    print(f"\n>> Unified District Master Feature Matrix: {master_district_df.shape}")
    print(f"   Saved to: {dist_master_parquet.relative_to(PROJECT_ROOT)}")

    # ==========================================
    # 2. TIER 1: DISTRICT FEATURE ENGINEERING
    # ==========================================
    print("\n[Phase 2/4] Engineering District Composite Indices & Vulnerability Ratios...")
    df_dist_engineered = compute_district_features(master_district_df)
    eng_parquet = proc_dir / "district_engineered_features.parquet"
    eng_csv = proc_dir / "district_engineered_features.csv"
    df_dist_engineered.to_parquet(eng_parquet, index=False)
    df_dist_engineered.to_csv(eng_csv, index=False)
    print(f">> District Engineered Features Matrix: {df_dist_engineered.shape}")
    print(f"   Features created: {list(df_dist_engineered.columns[2:])}")
    print(f"   Saved to: {eng_parquet.relative_to(PROJECT_ROOT)}")

    # ==========================================
    # 3. TIER 2: METROPOLITAN PREPROCESSING
    # ==========================================
    print("\n[Phase 3/4] Processing Tier 2 Metropolitan Datasets...")
    metro_files = [
        ("TABLE1B16.xlsx", "ipc"),
        ("TABLE1B33.xlsx", "total"),
        ("TABLE2B24.xlsx", "murder"),
        ("TABLE3B13.xlsx", "women"),
        ("TABLE4B13.xlsx", "child"),
        ("TABLE5B63.xlsx", "juv_socio"),
        ("TABLE9B33.xlsx", "cyber"),
    ]

    for fname, prefix in metro_files:
        p = curated_metro_dir / fname
        if p.exists():
            df_m = clean_metro_generic_table(p, prefix=prefix)
            stem = p.stem
            df_m.to_parquet(proc_metro_dir / f"{stem}.parquet", index=False)
            df_m.to_csv(proc_metro_dir / f"{stem}.csv", index=False)
            print(f"  Exported: {stem:<20} -> Shape: {df_m.shape}")

    # Special crime-head demography table (Table 5B.4)
    p_5b4 = curated_metro_dir / "TABLE5B41.xlsx"
    if p_5b4.exists():
        df_5b4 = parse_metro_juvenile_crime_demography(p_5b4)
        df_5b4.to_parquet(proc_metro_dir / "TABLE5B41.parquet", index=False)
        df_5b4.to_csv(proc_metro_dir / "TABLE5B41.csv", index=False)
        print(f"  Exported: TABLE5B41            -> Shape: {df_5b4.shape}")

    # Build Master Metropolitan Trends
    master_metro_df = build_metro_master_trends(curated_metro_dir)
    metro_master_parquet = proc_dir / "metro_master_feature_matrix.parquet"
    metro_master_csv = proc_dir / "metro_master_feature_matrix.csv"
    master_metro_df.to_parquet(metro_master_parquet, index=False)
    master_metro_df.to_csv(metro_master_csv, index=False)
    print(f"\n>> Unified Metropolitan Master Matrix: {master_metro_df.shape}")
    print(f"   Saved to: {metro_master_parquet.relative_to(PROJECT_ROOT)}")

    # ==========================================
    # 4. TIER 2: METRO TRENDS & CAGRs
    # ==========================================
    print("\n[Phase 4/4] Engineering Metropolitan Multi-Year Trends & Efficiency...")
    df_metro_trends = compute_metro_features(master_metro_df)
    metro_trends_parquet = proc_dir / "metro_master_trends.parquet"
    metro_trends_csv = proc_dir / "metro_master_trends.csv"
    df_metro_trends.to_parquet(metro_trends_parquet, index=False)
    df_metro_trends.to_csv(metro_trends_csv, index=False)
    print(f">> Metropolitan Engineered Trends: {df_metro_trends.shape}")
    print(f"   Metrics created: {list(df_metro_trends.columns[1:])}")
    print(f"   Saved to: {metro_trends_parquet.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
