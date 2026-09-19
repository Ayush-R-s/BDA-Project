"""
Metropolitan Analytics Module for NCRB 2024 Crime Analytics Suite.

Implements Policing Efficiency Quadrant Analysis, 3-Year CAGR momentum rankings,
and qualitative motive decomposition for murder and cybercrime.
"""

from typing import Dict, List, Tuple
import pandas as pd


def compute_policing_quadrants(df_trends: pd.DataFrame) -> pd.DataFrame:
    """Classify 37 metropolitan cities into 4 policing efficiency quadrants based on median splits."""
    df = df_trends.copy()

    med_crime_rate = df["crime_rate_per_lakh"].median()
    med_chargesheet = df["chargesheeting_rate"].median()

    def assign_quadrant(row):
        high_crime = row["crime_rate_per_lakh"] >= med_crime_rate
        high_cs = row["chargesheeting_rate"] >= med_chargesheet

        if high_crime and not high_cs:
            return "Q1: Critical Bottleneck (High Crime, Low Chargesheet)"
        elif high_crime and high_cs:
            return "Q2: High Incident / Active Enforcement (High Crime, High Chargesheet)"
        elif not high_crime and high_cs:
            return "Q3: High Containment Efficiency (Low Crime, High Chargesheet)"
        else:
            return "Q4: Low Reporting / Latent Risk (Low Crime, Low Chargesheet)"

    df["policing_quadrant"] = df.apply(assign_quadrant, axis=1)
    df["med_crime_rate"] = med_crime_rate
    df["med_chargesheet_rate"] = med_chargesheet
    return df


def extract_dominant_motives(df_master_metro: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose murder motives and cybercrime motives across major cities."""
    # 1. Murder Motives
    murder_cols = [c for c in df_master_metro.columns if c.startswith("murder_") and "total" not in c and "sl" not in c]
    murder_df = df_master_metro[["city"] + murder_cols].copy()
    murder_totals = murder_df[murder_cols].sum(axis=0).sort_values(ascending=False)

    # Clean up column labels for display
    clean_murder_summary = pd.DataFrame({
        "motive": [c.replace("murder_", "").replace("_", " ").title() for c in murder_totals.index],
        "total_incidents": murder_totals.values,
    })

    # 2. Cybercrime Motives
    cyber_cols = [c for c in df_master_metro.columns if c.startswith("cyber_") and "total" not in c and "sl" not in c]
    cyber_df = df_master_metro[["city"] + cyber_cols].copy()
    cyber_totals = cyber_df[cyber_cols].sum(axis=0).sort_values(ascending=False)

    clean_cyber_summary = pd.DataFrame({
        "motive": [c.replace("cyber_", "").replace("_", " ").title() for c in cyber_totals.index],
        "total_incidents": cyber_totals.values,
    })

    return clean_murder_summary, clean_cyber_summary
