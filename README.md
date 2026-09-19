# Big Data Analytics (BDA) Project: NCRB 2024 Crime Intelligence & Geospatial Analytics Suite

An end-to-end Big Data Analytics framework for analyzing, profiling, and modeling national crime patterns across Indian administrative districts and metropolitan cities using the National Crime Records Bureau (NCRB) 2024 publication.

---

## 1. Executive Summary & Problem Statement

Crime analytics across India presents massive challenges of heterogeneity, reporting variance, and dimensional scale. With over 800 police districts and 34+ million-plus metropolitan agglomerations, raw crime registries often exhibit high skew, disparate reporting practices, and complex multi-level structures.

This project implements a **Dual-Tier Analytical Architecture** that decouples national district-level spatial analysis from deep metropolitan trend dynamics:
1. **Tier 1 (Core District Backbone - 9 Files)**: All-India geospatial analysis covering ~1,000+ administrative units to identify violent crime hotspots, illegal firearm/contraband corridors, vulnerable population exploitation (women, children, SC/ST), youth delinquency, digital crime diffusion, and missing person proxies.
2. **Tier 2 (Metropolitan Comparative Layer - 8 Files)**: Multi-year trend modeling (2022–2024), policing efficiency benchmarking (crime rate vs. chargesheeting rate), causal homicide & cybercrime motive decomposition, and socio-demographic drivers of juvenile delinquency across India's largest urban clusters.

---

## 2. Project Architecture

```
                                    NCRB 2024 ANALYTICAL SUITE
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         │                                             │
                 DISTRICT CORE (Tier 1)                       METROPOLITAN (Tier 2)
              (Geospatial & Demographics)                  (Trends, Motives & Drivers)
                         │                                             │
             9 Curated Unified Datasets                    8 Curated Trend & Motive Datasets
                         │                                             │
             ├─ 1District: IPC/BNS Total Crimes            ├─ Table 1B.1: 3-Year IPC Trend & Chargesheet
             ├─ 2District: Special & Local Laws (SLL)      ├─ Table 1B.3: Total Crime Burden & State Share
             ├─ 3District: Crimes Against Women            ├─ Table 2B.2: 35 Murder Motives & Drivers
             ├─ 4District: Crimes Against Children         ├─ Table 3B.1: 3-Year Women Crime Trends
             ├─ 5District: Atrocities Against SCs          ├─ Table 4B.1: 3-Year Child Crime Trends
             ├─ 6District: Crimes Against STs              ├─ Table 5B.4: Juvenile Demography (Age/Gender)
             ├─ 7District: Juvenile IPC Crimes             ├─ Table 5B.6: Socioeconomic Determinants
             ├─ 9District: Cybercrimes (IT Act & IPC)      └─ Table 9B.3: Cybercrime Motives Decomposition
             └─ 10District: Missing Persons Demography
                         │                                             │
                         └──────────────────────┬──────────────────────┘
                                                │
                                    FEATURE ENGINEERING PIPELINE
                                                │
                         ┌──────────────────────┼──────────────────────┐
                         │                      │                      │
                  SPATIAL CLUSTERING        TIME-SERIES           CORRELATION &
                  (K-Means, DBSCAN)         FORECASTING         VULNERABILITY INDEX
                         │                      │                      │
                         └──────────────────────┼──────────────────────┘
                                                │
                                    INTERACTIVE DASHBOARD &
                                     POLICY RECOMMENDATIONS
```

---

## 3. Final Curated Dataset Portfolio Manifest

The portfolio has been verified and curated to eliminate data redundancy, remove non-standard formats (PDFs), and resolve temporal discrepancies (e.g. dropping unverified 2023 SLL juvenile labels):

### Tier 1: Core District Datasets (All-India Geospatial Coverage)

| File ID | File Name | Domain | Primary Dimensions | Key Analytical Question |
| :--- | :--- | :--- | :--- | :--- |
| **DIST_01_IPC** | `1DistrictwiseIPCCrimes2024.xlsx` | IPC / BNS Crimes | 156 crime categories (Violent, Property, Bodily, Theft, Murder) | Which districts represent severe nationwide crime hotspots across violent vs. property crimes? |
| **DIST_02_SLL** | `2DistrictwiseSLLCrimes2024.xlsx` | Special & Local Laws | 95 regulatory categories (Arms Act, NDPS, POCSO, Excise, Gambling) | How does illegal firearms and narcotics enforcement correlate with violent crime? |
| **DIST_03_WOMEN** | `3DistrictwiseCrimeagainstWomen2024.xlsx` | Women Safety | Rape, harassment, domestic cruelty (Sec 498A), dowry, age splits | What are the critical geographic clusters and reporting disparities of gender-based violence? |
| **DIST_04_CHILDREN**| `4DistrictwiseCrimeagainstChildren2024.xlsx` | Child Protection | POCSO, child trafficking, kidnapping, procuration of minors | Which district typologies exhibit disproportionate vulnerabilities in child exploitation? |
| **DIST_05_SC** | `5DistrictwiseCrimeagainstSCs2024.xlsx` | Marginalized Groups: SCs| PoA Act offenses, assault, land dispossession against Scheduled Castes | Are caste atrocities structurally isolated in specific agrarian corridors or uniformly spread? |
| **DIST_06_ST** | `6DistrictwiseCrimeagainstSTs2024.xlsx` | Marginalized Groups: STs| PoA Act crimes against Scheduled Tribes in tribal belts | How do crime profiles in Fifth/Sixth Schedule tribal belts differ from non-tribal districts? |
| **DIST_07_JUV_IPC**| `7DistrictwiseIPCCrimebyJuveniles2024.xlsx`| Youth Delinquency | 156 IPC offenses committed by juveniles in conflict with law | What categories of crime (property vs bodily) dominate youth delinquency across districts? |
| **DIST_09_CYBER** | `9DistrictwiseCyberCrimes2024.xlsx` | Digital Crime | 60 IT Act & cyber fraud heads across all districts | Is cybercrime confined to tech hubs or actively diffusing into rural administrative districts? |
| **DIST_10_MISSING**| `10DistrictwiseMissingPersons2024.xlsx` | Social Vulnerability | Missing children & adults by gender and age (proxy for trafficking) | Does missing person volume statistically predict kidnapping and human trafficking indicators? |

### Tier 2: Metropolitan Analytical Layer (~34+ Million-Plus Cities)

| File ID | File Name | Table Code | Temporal Range | Key Analytical Dimension |
| :--- | :--- | :--- | :--- | :--- |
| **METRO_01_TREND** | `TABLE1B16.xlsx` | Table 1B.1 | 2022–2024 | 3-Year IPC trends, estimated population, crime rates, chargesheeting efficiency. |
| **METRO_03_BURDEN** | `TABLE1B33.xlsx` | Table 1B.3 | 2022–2024 | Combined IPC+SLL macro crime rate and metropolitan percentage share of state totals. |
| **METRO_02_MURDER** | `TABLE2B24.xlsx` | Table 2B.2 | 2024 | Causal decomposition across 35 murder motives (disputes, greed, illicit affairs, feuds). |
| **METRO_04_WOMEN** | `TABLE3B13.xlsx` | Table 3B.1 | 2022–2024 | 3-Year crime against women trends, female population, and chargesheeting rates. |
| **METRO_05_CHILD** | `TABLE4B13.xlsx` | Table 4B.1 | 2022–2024 | 3-Year crime against children trends and comparative metropolitan child risk rate. |
| **METRO_06_JUV_DEMO**| `TABLE5B41.xlsx` | Table 5B.4 | 2024 | Juveniles apprehended by crime severity, age bracket (<12, 12–16, 16–18), and gender. |
| **METRO_07_JUV_SOCIO**| `TABLE5B63.xlsx` | Table 5B.6 | 2024 | Socioeconomic determinants: education level and family status (homeless/with parents). |
| **METRO_08_CYBER** | `TABLE9B33.xlsx` | Table 9B.3 | 2024 | 25 Cybercrime motives (financial fraud, extortion, personal revenge, harassment). |

---

## 4. Directory Structure

```
BDA Project/
├── data/
│   ├── raw/                           # Original immutable raw NCRB files
│   │   ├── Core/                      # Raw district files
│   │   ├── 2022-2024/                 # Raw multi-year tables
│   │   └── 2024/                      # Raw 2024 single-year tables
│   ├── curated/                       # Curated portfolio (Zero duplicates, audited)
│   │   ├── district/                  # 9 Tier 1 District Excel files
│   │   └── metropolitan/              # 8 Tier 2 Metropolitan Excel files
│   ├── processed/                     # Cleaned, standardized Parquet/CSV exports
│   ├── curated_portfolio_manifest.json# Machine-readable dataset catalog
│   └── curated_portfolio_manifest.csv # Tabular dataset manifest
├── notebooks/
│   └── 01_data_inventory.ipynb       # Portfolio audit, ingestion verification, EDA
├── src/
│   └── curated_loader.py              # Automated verification & header cleaning parsers
├── outputs/                           # Generated charts, cluster maps, and reports
├── .venv/                             # Python 3.13 virtual environment
└── README.md                          # Project documentation
```

---

## 5. Machine Learning & Analytical Roadmap

1. **Feature Engineering & Dimensionality Reduction**:
   - Aggregate district crime counts into composite indices:
     - **Violent Crime Index (VCI)**
     - **Property Crime Index (PCI)**
     - **Women Vulnerability Index (WVI)**
     - **Child Exploitation Index (CEI)**
     - **Digital Crime Intensity (DCI)**
   - Principal Component Analysis (PCA) to extract latent crime vulnerability dimensions.
2. **Unsupervised Spatial Clustering (Districts)**:
   - **K-Means & Hierarchical Clustering**: Segmenting all ~800+ districts into risk typologies (e.g., Agrarian Atrocity Belts, Urban Theft Hubs, Contraband Transit Corridors, Low-Crime Stable Zones).
   - **DBSCAN Density-Based Clustering**: Detecting anomalous high-intensity crime clusters and border smuggling outliers.
3. **Metropolitan Multi-Year Dynamic Modeling**:
   - **CAGR Trend Forecasting**: 3-year trajectory analysis for violent crime vs chargesheeting rate.
   - **Policing Efficiency Quadrant Matrix**: Mapping cities on *Crime Incidence Rate* vs *Chargesheeting Rate* to identify overburdened vs high-performing law enforcement jurisdictions.
4. **Socio-Causal Driver Analysis**:
   - Correlation and multi-variate regression evaluating the link between juvenile educational dropout/family disruption and violent crime apprehension rates.

---

## 6. Quick Start & Ingestion

To run the automated portfolio validator:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run the automated verification script
python src/curated_loader.py
```

To run the interactive exploratory data inventory:
Open `notebooks/01_data_inventory.ipynb` in Jupyter Notebook / VS Code.
