"""
Feature Engineering Module for NCRB 2024 Crime Analytics Suite.

Constructs domain-specific composite indices, vulnerability ratios, and
growth rates for both administrative districts and metropolitan cities.
"""

import numpy as np
import pandas as pd


def compute_district_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer composite indices and vulnerability ratios from the master district matrix."""
    feats = pd.DataFrame()
    feats["state"] = df["state"]
    feats["district"] = df["district"]

    # 1. Total Volume Aggregations
    ipc_numeric = [c for c in df.columns if c.startswith("ipc_") and pd.api.types.is_numeric_dtype(df[c])]
    sll_numeric = [c for c in df.columns if c.startswith("sll_") and pd.api.types.is_numeric_dtype(df[c])]
    women_numeric = [c for c in df.columns if c.startswith("women_") and pd.api.types.is_numeric_dtype(df[c])]
    child_numeric = [c for c in df.columns if c.startswith("child_") and pd.api.types.is_numeric_dtype(df[c])]
    sc_numeric = [c for c in df.columns if c.startswith("sc_") and pd.api.types.is_numeric_dtype(df[c])]
    st_numeric = [c for c in df.columns if c.startswith("st_") and pd.api.types.is_numeric_dtype(df[c])]
    juv_numeric = [c for c in df.columns if c.startswith("juv_ipc_") and pd.api.types.is_numeric_dtype(df[c])]
    cyber_numeric = [c for c in df.columns if c.startswith("cyber_") and pd.api.types.is_numeric_dtype(df[c])]
    missing_numeric = [c for c in df.columns if c.startswith("missing_") and pd.api.types.is_numeric_dtype(df[c])]

    # Compute domain totals (using specific total columns if present or sum of leaf heads)
    # IPC Total
    ipc_tot_cols = [c for c in ipc_numeric if "total" in c and "col" in c]
    if ipc_tot_cols:
        feats["total_ipc_crimes"] = df[ipc_tot_cols[-1]]
    else:
        feats["total_ipc_crimes"] = df[ipc_numeric].sum(axis=1)

    # If any district reports 0 in total_ipc_crimes, fallback to sum of columns
    mask_zero = feats["total_ipc_crimes"] == 0
    if mask_zero.any():
        feats.loc[mask_zero, "total_ipc_crimes"] = df.loc[mask_zero, ipc_numeric].sum(axis=1)

    feats["total_sll_crimes"] = df[sll_numeric].sum(axis=1)
    feats["total_crime_burden"] = feats["total_ipc_crimes"] + feats["total_sll_crimes"]

    # 2. Violent Crime Subtotal (Murder, Attempt, Rape, Kidnapping, Riot)
    violent_keywords = ["murder", "rape", "kidnapping", "riot", "dacoity", "culpable_homicide"]
    violent_cols = [c for c in ipc_numeric if any(k in c for k in violent_keywords) and "total" not in c]
    feats["violent_crime_total"] = df[violent_cols].sum(axis=1) if violent_cols else 0

    # 3. Property Crime Subtotal (Theft, Burglary, Robbery, Criminal Breach of Trust, Cheating)
    property_keywords = ["theft", "burglary", "robbery", "criminal_breach", "cheating", "extortion"]
    property_cols = [c for c in ipc_numeric if any(k in c for k in property_keywords) and "total" not in c]
    feats["property_crime_total"] = df[property_cols].sum(axis=1) if property_cols else 0

    # 4. Weapons & Narcotics Enforcements
    arms_cols = [c for c in sll_numeric if "arms" in c and "total" not in c]
    ndps_cols = [c for c in sll_numeric if "ndps" in c or "narcotic" in c or "drugs" in c]
    feats["arms_act_cases"] = df[arms_cols].sum(axis=1) if arms_cols else 0
    feats["ndps_narcotics_cases"] = df[ndps_cols].sum(axis=1) if ndps_cols else 0
    feats["contraband_enforcement_total"] = feats["arms_act_cases"] + feats["ndps_narcotics_cases"]

    # 5. Vulnerable Groups
    feats["crimes_against_women_total"] = df[women_numeric].sum(axis=1)
    feats["crimes_against_children_total"] = df[child_numeric].sum(axis=1)
    feats["crimes_against_sc_total"] = df[sc_numeric].sum(axis=1)
    feats["crimes_against_st_total"] = df[st_numeric].sum(axis=1)
    feats["sc_st_atrocities_total"] = feats["crimes_against_sc_total"] + feats["crimes_against_st_total"]

    # 6. Youth Delinquency & Cybercrime
    feats["juvenile_ipc_crimes_total"] = df[juv_numeric].sum(axis=1)
    feats["cybercrimes_total"] = df[cyber_numeric].sum(axis=1)

    # 7. Missing Persons & Trafficking Proxies
    missing_tot_cols = [c for c in missing_numeric if "total" in c]
    feats["missing_persons_total"] = df[missing_tot_cols].sum(axis=1) if missing_tot_cols else df[missing_numeric].sum(axis=1)
    missing_child_cols = [c for c in missing_numeric if "children" in c or "below" in c]
    feats["missing_children_total"] = df[missing_child_cols].sum(axis=1) if missing_child_cols else 0

    # 8. Normalized Shares and Composite Ratios (avoid division by zero with denominator offset)
    denom = feats["total_ipc_crimes"].replace(0, 1)

    feats["violent_crime_ratio"] = np.round(feats["violent_crime_total"] / denom, 4)
    feats["property_crime_ratio"] = np.round(feats["property_crime_total"] / denom, 4)
    feats["women_vulnerability_share"] = np.round(feats["crimes_against_women_total"] / denom, 4)
    feats["child_vulnerability_share"] = np.round(feats["crimes_against_children_total"] / denom, 4)
    feats["caste_atrocity_share"] = np.round(feats["sc_st_atrocities_total"] / denom, 4)
    feats["juvenile_delinquency_ratio"] = np.round(feats["juvenile_ipc_crimes_total"] / denom, 4)
    feats["cyber_crime_intensity_per_1k"] = np.round((feats["cybercrimes_total"] * 1000.0) / denom, 4)

    # Missing children per kidnapping case
    kidnap_cols = [c for c in child_numeric if "kidnapping" in c]
    kidnap_denom = df[kidnap_cols].sum(axis=1).replace(0, 1) if kidnap_cols else pd.Series(1, index=df.index)
    feats["trafficking_vulnerability_proxy"] = np.round(feats["missing_children_total"] / kidnap_denom, 4)

    return feats


def compute_metro_features(df_metro: pd.DataFrame) -> pd.DataFrame:
    """Engineer multi-year trend growth rates, CAGR, and efficiency metrics for metropolitan cities."""
    feats = pd.DataFrame()
    feats["city"] = df_metro["city"]

    # 1. Multi-Year IPC Crime Trends (2022, 2023, 2024)
    c22 = "ipc_2022" if "ipc_2022" in df_metro.columns else [c for c in df_metro.columns if "2022" in c][0]
    c23 = "ipc_2023" if "ipc_2023" in df_metro.columns else [c for c in df_metro.columns if "2023" in c][0]
    c24 = "ipc_2024" if "ipc_2024" in df_metro.columns else [c for c in df_metro.columns if "2024" in c][0]

    feats["ipc_crimes_2022"] = df_metro[c22]
    feats["ipc_crimes_2023"] = df_metro[c23]
    feats["ipc_crimes_2024"] = df_metro[c24]

    # Compute 3-Year CAGR for IPC Crimes
    # CAGR = (V_final / V_begin)**(1/2) - 1
    safe_begin = feats["ipc_crimes_2022"].replace(0, np.nan)
    feats["ipc_cagr_2022_2024"] = np.round(
        ((feats["ipc_crimes_2024"] / safe_begin) ** 0.5 - 1) * 100, 2
    ).fillna(0.0)

    # 2. Population and Rates
    pop_cols = [c for c in df_metro.columns if "population" in c]
    if pop_cols:
        feats["population_lakhs"] = df_metro[pop_cols[0]]

    rate_cols = [c for c in df_metro.columns if "rate" in c and "charge" not in c and "ipc" in c]
    if rate_cols:
        feats["crime_rate_per_lakh"] = df_metro[rate_cols[0]]

    chargesheet_cols = [c for c in df_metro.columns if "chargesheet" in c]
    if chargesheet_cols:
        feats["chargesheeting_rate"] = df_metro[chargesheet_cols[0]]

    # 3. Policing Efficiency Index (Chargesheeting Rate / Crime Rate)
    if "chargesheeting_rate" in feats.columns and "crime_rate_per_lakh" in feats.columns:
        safe_rate = feats["crime_rate_per_lakh"].replace(0, 1)
        feats["policing_efficiency_index"] = np.round(feats["chargesheeting_rate"] / safe_rate, 4)

    # 4. Multi-Year Women Crime Trends if available
    women_22 = [c for c in df_metro.columns if "women" in c and "2022" in c]
    women_24 = [c for c in df_metro.columns if "women" in c and "2024" in c]
    if women_22 and women_24:
        feats["women_crimes_2022"] = df_metro[women_22[0]]
        feats["women_crimes_2024"] = df_metro[women_24[0]]
        w_safe = feats["women_crimes_2022"].replace(0, np.nan)
        feats["women_crime_cagr"] = np.round(((feats["women_crimes_2024"] / w_safe) ** 0.5 - 1) * 100, 2).fillna(0.0)

    # 5. Multi-Year Child Crime Trends if available
    child_22 = [c for c in df_metro.columns if "child" in c and "2022" in c]
    child_24 = [c for c in df_metro.columns if "child" in c and "2024" in c]
    if child_22 and child_24:
        feats["child_crimes_2022"] = df_metro[child_22[0]]
        feats["child_crimes_2024"] = df_metro[child_24[0]]
        c_safe = feats["child_crimes_2022"].replace(0, np.nan)
        feats["child_crime_cagr"] = np.round(((feats["child_crimes_2024"] / c_safe) ** 0.5 - 1) * 100, 2).fillna(0.0)

    return feats
