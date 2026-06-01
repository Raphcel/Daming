"""
===============================================================================
ASEAN Temperature Anomaly Prediction - Focused Time Series & Deep Learning Pipeline
===============================================================================
Topik: Implementasi teknik data mining untuk analisis data climate change
       dan perubahan tutupan lahan

Fokus utama:
  a. Time Series Data Mining (ADF, decomposition, ACF/PACF, supervised forecasting)
  b. Deep Learning (LSTM, GRU - PyTorch)

Model pembanding:
  Baseline, Ridge, Random Forest Regressor, HistGradientBoosting Regressor

Dataset: FAOSTAT (FAO) - Temperature Change, Land Cover, Emissions
===============================================================================
"""
from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["font.size"] = 10

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
ASEAN_MEMBERS = [
    "Brunei Darussalam",
    "Cambodia",
    "Indonesia",
    "Lao People's Democratic Republic",
    "Malaysia",
    "Myanmar",
    "Philippines",
    "Singapore",
    "Thailand",
    "Viet Nam",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "faostat"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_TEMP = (
    RAW_DATA_DIR
    / "Environment_Temperature_change_E_All_Data"
    / "Environment_Temperature_change_E_All_Data_NOFLAG.csv"
)
RAW_LAND = (
    RAW_DATA_DIR
    / "Environment_LandCover_E_All_Data"
    / "Environment_LandCover_E_All_Data_NOFLAG.csv"
)
RAW_EMISSIONS = (
    RAW_DATA_DIR
    / "Emissions_Totals_E_All_Data"
    / "Emissions_Totals_E_All_Data_NOFLAG.csv"
)
ASEAN_CSV = PROCESSED_DATA_DIR / "merged_climate_landcover_asean.csv"


def artifact_path(output_dir: Path, filename: str) -> Path:
    """Return the organized output path for a generated artifact."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".png":
        path = output_dir / "figures" / filename
    elif suffix == ".csv":
        path = output_dir / "tables" / filename
    elif suffix == ".json":
        path = output_dir / "reports_data" / filename
    else:
        path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs(output_dir: Path = OUTPUT_DIR) -> None:
    for dirname in ("figures", "tables", "reports_data", "report_render"):
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)

# Temperature discretization bins
TEMP_BINS = [-np.inf, 0.5, 1.0, 1.5, np.inf]
TEMP_LABELS = ["Rendah", "Sedang", "Tinggi", "Kritis"]

# Emission change discretization
CHANGE_BINS = [-np.inf, -0.01, 0.01, np.inf]
CHANGE_LABELS = ["Turun", "Stabil", "Naik"]


@dataclass(frozen=True)
class ModelingConfig:
    train_end_year: int = 2016
    validation_end_year: int = 2019
    max_feature_missing: float = 0.95
    sequence_window: int = 5
    random_seed: int = 42
    deep_epochs: int = 30
    deep_patience: int = 6
    deep_batch_size: int = 256
    deep_weight_decay: float = 1e-4
    deep_dropout: float = 0.05
    deep_tuning_windows: tuple = (3, 5, 7)
    deep_tuning_hidden_dims: tuple = (16, 24, 48)
    deep_tuning_learning_rates: tuple = (0.001, 0.003)
    deep_tuning_dropouts: tuple = (0.05, 0.15)
    deep_tuning_num_layers: tuple = (1, 2)
    deep_tuning_weight_decays: tuple = (1e-4, 1e-3)
    scenario_end_year: int = 2030
    n_clusters_range: tuple = (2, 7)
    top_n_features: int = 30


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


# =============================================================================
# SECTION 1: DATA LOADING & MERGE
# =============================================================================

def year_columns(df: pd.DataFrame, start: int = 1992, end: int = 2022) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if isinstance(col, str) and col.startswith("Y") and col[1:].isdigit():
            year = int(col[1:])
            if start <= year <= end:
                cols.append(col)
    return cols


def melt_years(df, id_vars, value_name, start=1992, end=2022):
    cols = year_columns(df, start=start, end=end)
    long_df = df[id_vars + cols].melt(
        id_vars=id_vars, value_vars=cols,
        var_name="Year", value_name=value_name,
    )
    long_df["Year"] = long_df["Year"].str.replace("Y", "", regex=False).astype(int)
    return long_df


def load_faostat_panel(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Load and merge raw FAOSTAT files into a country-year panel."""
    print("=" * 70)
    print("TAHAP 1: MEMUAT DAN MENGGABUNGKAN DATA FAOSTAT")
    print("=" * 70)

    temp = pd.read_csv(RAW_TEMP)
    land = pd.read_csv(RAW_LAND)
    emissions = pd.read_csv(RAW_EMISSIONS)

    # Temperature: Meteorological year, Temperature change
    temp = temp[
        (temp["Months"] == "Meteorological year")
        & (temp["Element"] == "Temperature change")
    ]
    temp_long = melt_years(temp, ["Area"], "Temperature_Change")
    print(f"  Temperature: {len(temp_long)} baris")

    # Land Cover: Area from CCI_LC
    land = land[land["Element"].eq("Area from CCI_LC")]
    land_long = melt_years(land, ["Area", "Item"], "LandCover_Area")
    land_pivot = (
        land_long.pivot_table(
            index=["Area", "Year"], columns="Item",
            values="LandCover_Area", aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    print(f"  Land Cover: {len(land_pivot)} baris, {land_pivot.shape[1] - 2} kategori")

    # Emissions: CO2eq, deduplicate by source priority
    emissions = emissions[emissions["Element"].eq("Emissions (CO2eq) (AR5)")].copy()
    source_order = {"FAO TIER 1": 0, "UNFCCC": 1}
    emissions["source_rank"] = emissions["Source"].map(source_order).fillna(9)
    emissions_long = melt_years(
        emissions, ["Area", "Item", "Source", "source_rank"], "Emissions_CO2eq",
    )
    emissions_long = emissions_long.sort_values("source_rank")
    emissions_long = emissions_long.drop_duplicates(
        subset=["Area", "Year", "Item"], keep="first",
    )
    emissions_pivot = (
        emissions_long.pivot_table(
            index=["Area", "Year"], columns="Item",
            values="Emissions_CO2eq", aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    print(f"  Emissions: {len(emissions_pivot)} baris, {emissions_pivot.shape[1] - 2} kategori")

    # Merge
    panel = temp_long.merge(land_pivot, on=["Area", "Year"], how="inner")
    panel = panel.merge(emissions_pivot, on=["Area", "Year"], how="inner")
    panel = panel.sort_values(["Area", "Year"]).reset_index(drop=True)

    output_dir.mkdir(exist_ok=True)
    panel.to_csv(artifact_path(output_dir, "faostat_country_year_panel.csv"), index=False)

    print(f"\n  Panel akhir: {panel.shape[0]} baris x {panel.shape[1]} kolom")
    print(f"  Negara unik: {panel['Area'].nunique()}")
    print(f"  Rentang tahun: {panel['Year'].min()} - {panel['Year'].max()}")
    return panel


# =============================================================================
# SECTION 2: ENHANCED EDA
# =============================================================================

def run_enhanced_eda(panel: pd.DataFrame, output_dir: Path) -> dict:
    """Enhanced EDA: stationarity, correlation, VIF, lag analysis."""
    print("\n" + "=" * 70)
    print("TAHAP 2: ANALISIS DATA EKSPLORATIF LANJUTAN")
    print("=" * 70)

    results = {}

    # --- 2.1 Missing Value Analysis ---
    print("\n--- 2.1 Analisis Missing Value ---")
    missing = panel.isnull().sum()
    missing_pct = (missing / len(panel) * 100).round(2)
    missing_df = pd.DataFrame({"Jumlah_Missing": missing, "Persentase": missing_pct})
    missing_df = missing_df[missing_df["Jumlah_Missing"] > 0].sort_values("Persentase", ascending=False)
    print(f"  Kolom dengan missing: {len(missing_df)} dari {panel.shape[1]}")
    print(missing_df.head(10).to_string())
    missing_df.to_csv(artifact_path(output_dir, "missing_value_summary.csv"))
    results["missing_summary"] = missing_df

    # --- 2.2 ADF Stationarity Test ---
    print("\n--- 2.2 Uji Stasioneritas (Augmented Dickey-Fuller) ---")
    adf_results = []
    asean_panel = panel[panel["Area"].isin(ASEAN_MEMBERS)]
    for country in sorted(asean_panel["Area"].unique()):
        series = asean_panel[asean_panel["Area"] == country]["Temperature_Change"].dropna()
        if len(series) < 8:
            continue
        try:
            stat, pval, *_ = adfuller(series, maxlag=5)
            stationary = "Stasioner" if pval < 0.05 else "Non-Stasioner"
            adf_results.append({
                "Negara": country, "ADF_Statistic": round(stat, 4),
                "p_value": round(pval, 4), "Kesimpulan": stationary,
            })
        except Exception:
            continue

    adf_df = pd.DataFrame(adf_results)
    print(adf_df.to_string(index=False))
    adf_df.to_csv(artifact_path(output_dir, "adf_stationarity_results.csv"), index=False)
    results["adf_tests"] = adf_df

    n_nonstationary = (adf_df["Kesimpulan"] == "Non-Stasioner").sum()
    print(f"\n  Interpretasi: {n_nonstationary}/{len(adf_df)} negara ASEAN memiliki")
    print("  series suhu non-stasioner, mengkonfirmasi tren pemanasan yang perlu")
    print("  ditangani dalam modeling (differencing atau detrending).")

    # --- 2.3 Correlation Heatmap ---
    print("\n--- 2.3 Korelasi Fitur Utama (ASEAN) ---")
    key_features = [
        "Temperature_Change", "Tree-covered areas", "Herbaceous crops",
        "Artificial surfaces (including urban and associated areas)",
        "AFOLU", "Energy", "LULUCF", "Net Forest conversion",
        "Forest fires", "Agrifood systems",
    ]
    available = [f for f in key_features if f in asean_panel.columns]
    corr_matrix = asean_panel[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Korelasi Fitur Utama - Negara ASEAN", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(artifact_path(output_dir, "correlation_heatmap_asean.png"))
    plt.close()
    print("  Heatmap disimpan.")
    results["correlation_matrix"] = corr_matrix

    # Identify highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            r = abs(corr_matrix.iloc[i, j])
            if r > 0.85:
                high_corr_pairs.append(
                    (corr_matrix.index[i], corr_matrix.columns[j], round(r, 3))
                )
    if high_corr_pairs:
        print("  Pasangan dengan korelasi tinggi (|r| > 0.85):")
        for a, b, r in high_corr_pairs:
            print(f"    {a} <-> {b}: {r}")
    pd.DataFrame(
        high_corr_pairs, columns=["feature_a", "feature_b", "abs_correlation"]
    ).to_csv(artifact_path(output_dir, "high_correlation_pairs_asean.csv"), index=False)

    # --- 2.4 Temporal Trend ---
    print("\n--- 2.4 Tren Temporal Anomali Suhu ---")
    asean_yearly = (
        asean_panel.groupby("Year")["Temperature_Change"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(
        asean_yearly["Year"],
        asean_yearly["mean"] - asean_yearly["std"],
        asean_yearly["mean"] + asean_yearly["std"],
        alpha=0.2, color="#3b82f6",
    )
    ax.plot(asean_yearly["Year"], asean_yearly["mean"], "o-", color="#3b82f6", linewidth=2)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Anomali Suhu (deg C)")
    ax.set_title("Tren Rata-rata Anomali Suhu ASEAN (1992-2022)", fontweight="bold")
    # Linear regression overlay
    valid = asean_yearly.dropna(subset=["mean"])
    slope, intercept, r_val, _, _ = stats.linregress(valid["Year"], valid["mean"])
    ax.plot(valid["Year"], intercept + slope * valid["Year"], "--", color="red",
            label=f"Tren linear: +{slope:.3f} deg C/tahun (R2={r_val**2:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(artifact_path(output_dir, "asean_temperature_trend.png"))
    plt.close()
    print(f"  Tren linear: +{slope:.4f} deg C/tahun (R2={r_val**2:.4f})")
    results["temp_trend_slope"] = slope

    # --- 2.5 Descriptive Statistics ---
    print("\n--- 2.5 Statistik Deskriptif ---")
    desc = panel.describe().T
    desc.to_csv(artifact_path(output_dir, "descriptive_statistics.csv"))
    print(f"  Disimpan ke descriptive_statistics.csv")

    return results


# =============================================================================
# SECTION 3: PREPROCESSING
# =============================================================================

def preprocess_panel(panel: pd.DataFrame, config: ModelingConfig) -> pd.DataFrame:
    """Full preprocessing: missing values, transformation, discretization."""
    print("\n" + "=" * 70)
    print("TAHAP 3: PREPROCESSING DATA")
    print("=" * 70)
    df = panel.copy()

    # --- 3.1 Drop columns with >95% missing ---
    print("\n--- 3.1 Penanganan Missing Value (3-tier) ---")
    missing_pct = df.isnull().mean()
    drop_cols = missing_pct[missing_pct > config.max_feature_missing].index.tolist()
    if drop_cols:
        print(f"  Tier 1 - Drop kolom (>{config.max_feature_missing*100:.0f}% missing): {drop_cols}")
        df = df.drop(columns=drop_cols)

    # --- 3.2 Per-country interpolation ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Year"]
    for col in numeric_cols:
        df[col] = df.groupby("Area")[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
    remaining_after_interp = df[numeric_cols].isnull().sum().sum()
    print(f"  Tier 2 - Interpolasi per negara: sisa missing = {remaining_after_interp}")

    # --- 3.3 Global median imputation ---
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    remaining_after_median = df[numeric_cols].isnull().sum().sum()
    print(f"  Tier 3 - Imputasi median global: sisa missing = {remaining_after_median}")

    # --- 3.4 Log-transform highly skewed features ---
    print("\n--- 3.2 Transformasi Data ---")
    skewed_candidates = [
        "Fires in organic soils", "Fires in humid tropical forests",
        "Forest fires", "Net Forest conversion",
    ]
    log_transformed = []
    for col in skewed_candidates:
        if col in df.columns:
            skewness = df[col].skew()
            if abs(skewness) > 2:
                df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))
                log_transformed.append((col, round(skewness, 2)))
    if log_transformed:
        print(f"  Log-transform diterapkan pada: {[x[0] for x in log_transformed]}")
        print(f"  (Skewness awal: {log_transformed})")

    # --- 3.5 Feature engineering (deltas, pct_change) ---
    df = df.sort_values(["Area", "Year"]).copy()
    key_cols = [
        "Tree-covered areas",
        "Artificial surfaces (including urban and associated areas)",
        "Herbaceous crops", "AFOLU", "Agrifood systems",
        "Energy", "LULUCF", "Net Forest conversion", "Forest fires",
    ]
    for col in key_cols:
        if col in df.columns:
            df[f"delta_{col}"] = df.groupby("Area")[col].diff()
            df[f"pct_change_{col}"] = (
                df.groupby("Area")[col]
                .pct_change(fill_method=None)
                .replace([np.inf, -np.inf], np.nan)
            )
    engineered = [c for c in df.columns if c.startswith(("delta_", "pct_change_"))]
    df[engineered] = df[engineered].fillna(0.0)
    print(f"  Fitur delta/pct_change ditambahkan: {len(engineered)} fitur baru")

    # --- 3.6 Discretization ---
    print("\n--- 3.3 Diskretisasi ---")
    df["Temp_Category"] = pd.cut(
        df["Temperature_Change"], bins=TEMP_BINS, labels=TEMP_LABELS
    )
    print(f"  Temperature_Change didiskretisasi ke: {TEMP_LABELS}")
    print(f"  Distribusi: {df['Temp_Category'].value_counts().to_dict()}")

    # Discretize feature changes for descriptive risk summaries if needed.
    for col in ["delta_AFOLU", "delta_Energy", "delta_LULUCF",
                 "delta_Net Forest conversion", "delta_Forest fires"]:
        if col in df.columns:
            col_std = df[col].std()
            if col_std > 0:
                thresh = col_std * 0.1
                df[f"cat_{col}"] = pd.cut(
                    df[col], bins=[-np.inf, -thresh, thresh, np.inf],
                    labels=CHANGE_LABELS,
                )

    # Discretize land cover changes
    for col in ["delta_Tree-covered areas", "delta_Herbaceous crops",
                 "delta_Artificial surfaces (including urban and associated areas)"]:
        if col in df.columns:
            col_std = df[col].std()
            if col_std > 0:
                thresh = col_std * 0.1
                df[f"cat_{col}"] = pd.cut(
                    df[col], bins=[-np.inf, -thresh, thresh, np.inf],
                    labels=CHANGE_LABELS,
                )

    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    print(f"  Fitur kategorikal tambahan: {len(cat_cols)} kolom")

    # --- 3.7 Feature selection note ---
    print("\n--- 3.4 Seleksi Fitur ---")
    numeric_features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"Year"} and not c.startswith("cat_")
    ]
    print(
        "  Korelasi tinggi akan diseleksi dari train split saja "
        "di feature_columns() untuk mengurangi leakage."
    )

    final_numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"Year"}
    ]
    print(f"  Fitur numerik final: {len(final_numeric)}")

    return df


# =============================================================================
# SECTION 4: SUPERVISED DATASET CONSTRUCTION
# =============================================================================

def make_supervised(panel: pd.DataFrame, config: ModelingConfig) -> pd.DataFrame:
    """Create supervised dataset: predict year-over-year temperature delta."""
    df = panel.copy()
    df["target_level_next_year"] = df.groupby("Area")["Temperature_Change"].shift(-1)
    df["target_next_year"] = df["target_level_next_year"] - df["Temperature_Change"]
    df["target_year"] = df["Year"] + 1
    df = df.dropna(subset=["target_level_next_year", "target_next_year"]).copy()

    # Label ASEAN membership
    if ASEAN_CSV.exists():
        asean = pd.read_csv(ASEAN_CSV)
        asean_pairs = set(zip(asean["Area"], asean["Year"]))
        df["is_asean"] = [
            (area, year) in asean_pairs
            for area, year in zip(df["Area"], df["Year"])
        ]
    else:
        df["is_asean"] = df["Area"].isin(ASEAN_MEMBERS)

    # Temporal split
    df["split"] = np.select(
        [df["Year"] <= config.train_end_year, df["Year"] <= config.validation_end_year],
        ["train", "validation"],
        default="test",
    )

    # Discretize target level for risk-category interpretation.
    df["target_category"] = pd.cut(
        df["target_level_next_year"], bins=TEMP_BINS, labels=TEMP_LABELS,
    )

    delta_check = (
        df["target_level_next_year"] - df["Temperature_Change"] - df["target_next_year"]
    ).abs().max()
    if pd.notna(delta_check) and delta_check > 1e-10:
        raise ValueError("Delta target validation failed.")

    return df.reset_index(drop=True)


def feature_columns(supervised, config):
    excluded = {"target_next_year", "target_level_next_year", "target_year", "split", "is_asean",
                "Temp_Category", "target_category"}
    candidates = [c for c in supervised.columns if c not in excluded]
    numeric = [
        c for c in candidates
        if c != "Area" and not c.startswith("cat_")
        and pd.api.types.is_numeric_dtype(supervised[c])
        and supervised[c].isna().mean() <= config.max_feature_missing
    ]
    train_mask = supervised["split"].eq("train")
    train_numeric = supervised.loc[train_mask, numeric]
    if len(numeric) > 1 and not train_numeric.empty:
        corr = train_numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        if to_drop:
            numeric = [col for col in numeric if col not in set(to_drop)]
    categorical = ["Area"]
    return numeric, categorical


def make_preprocessor(numeric, categorical):
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", encoder, categorical),
    ])


# =============================================================================
# SECTION 5c: DEEP LEARNING (LSTM & GRU)
# =============================================================================

class RecurrentRegressor(nn.Module):
    def __init__(
        self,
        model_type: str,
        input_dim: int,
        hidden_dim: int = 24,
        dropout: float = 0.05,
        num_layers: int = 1,
    ):
        super().__init__()
        rnn_cls = nn.GRU if model_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TorchSequenceModel:
    def __init__(
        self,
        model_type,
        numeric,
        categorical,
        config,
        sequence_window=None,
        hidden_dim=24,
        learning_rate=0.003,
        dropout=None,
        num_layers=1,
        batch_size=None,
        weight_decay=None,
        max_epochs=None,
        patience=None,
    ):
        self.model_type = model_type
        self.numeric = numeric
        self.categorical = categorical
        self.config = config
        self.sequence_window = int(sequence_window or config.sequence_window)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.dropout = float(config.deep_dropout if dropout is None else dropout)
        self.num_layers = int(num_layers)
        self.batch_size = int(config.deep_batch_size if batch_size is None else batch_size)
        self.weight_decay = float(config.deep_weight_decay if weight_decay is None else weight_decay)
        self.max_epochs = int(config.deep_epochs if max_epochs is None else max_epochs)
        self.patience_limit = int(config.deep_patience if patience is None else patience)
        self.preprocessor = make_preprocessor(numeric, categorical)
        self.model = None
        self.device = torch.device("cpu")
        self.best_epoch = None
        self.best_val_loss = np.nan

    def _sequences(self, supervised, transformed):
        seqs, targets, indices = [], [], []
        window = self.sequence_window
        transformed_df = pd.DataFrame(transformed, index=supervised.index)
        for _, group in supervised.sort_values(["Area", "Year"]).groupby("Area"):
            group = group.sort_values("Year")
            for pos in range(window - 1, len(group)):
                idx = group.index[pos]
                seq_idx = group.index[pos - window + 1: pos + 1]
                seqs.append(transformed_df.loc[seq_idx].to_numpy(dtype=np.float32))
                targets.append(float(group.loc[idx, "target_next_year"]))
                indices.append(int(idx))
        if not seqs:
            return (
                np.empty((0, window, transformed.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=int),
            )
        return np.stack(seqs), np.array(targets, dtype=np.float32), np.array(indices, dtype=int)

    def fit(self, supervised):
        features = self.numeric + self.categorical
        train_rows = supervised["split"].eq("train")
        self.preprocessor.fit(supervised.loc[train_rows, features])
        transformed = self.preprocessor.transform(supervised[features])
        x_seq, y_seq, idx_seq = self._sequences(supervised, transformed)
        train_mask = supervised.loc[idx_seq, "split"].eq("train").to_numpy()
        val_mask = supervised.loc[idx_seq, "split"].eq("validation").to_numpy()

        input_dim = x_seq.shape[-1]
        self.model = RecurrentRegressor(
            self.model_type, input_dim, hidden_dim=self.hidden_dim,
            dropout=self.dropout, num_layers=self.num_layers,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        train_dataset = TensorDataset(
            torch.tensor(x_seq[train_mask], dtype=torch.float32),
            torch.tensor(y_seq[train_mask], dtype=torch.float32),
        )
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_state, best_val, patience = None, float("inf"), 0
        train_losses, val_losses = [], []

        for epoch in range(self.max_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb.to(self.device))
                loss = loss_fn(pred, yb.to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_losses.append(epoch_loss / max(n_batches, 1))

            self.model.eval()
            with torch.no_grad():
                if val_mask.any():
                    val_pred = self.model(torch.tensor(x_seq[val_mask], dtype=torch.float32))
                    val_loss = loss_fn(val_pred, torch.tensor(y_seq[val_mask], dtype=torch.float32)).item()
                else:
                    val_loss = train_losses[-1]
            val_losses.append(val_loss)

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                self.best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= self.patience_limit:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.best_val_loss = float(best_val)
        return self

    def predict(self, supervised):
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        transformed = self.preprocessor.transform(
            supervised[self.numeric + self.categorical]
        )
        x_seq, _, idx_seq = self._sequences(supervised, transformed)
        pred = np.full(len(supervised), np.nan, dtype=float)
        if len(x_seq) == 0:
            return pred
        self.model.eval()
        with torch.no_grad():
            yhat = self.model(torch.tensor(x_seq, dtype=torch.float32)).cpu().numpy()
        pred[idx_seq] = yhat
        return pred


# =============================================================================
# SECTION 5e: TIME SERIES DATA MINING
# =============================================================================

def run_timeseries_mining(panel: pd.DataFrame, output_dir: Path) -> dict:
    """Time series decomposition, ACF/PACF analysis."""
    print("\n" + "=" * 70)
    print("TEKNIK (e): TIME SERIES DATA MINING")
    print("=" * 70)
    results = {}

    asean_data = panel[panel["Area"].isin(ASEAN_MEMBERS)]
    asean_mean = asean_data.groupby("Year")["Temperature_Change"].mean().dropna()

    # --- Seasonal Decomposition ---
    print("\n--- Dekomposisi Time Series (ASEAN Mean) ---")
    if len(asean_mean) >= 6:
        try:
            decomp = seasonal_decompose(asean_mean, model="additive", period=min(5, len(asean_mean) // 2))

            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            decomp.observed.plot(ax=axes[0], title="Observasi", color="#3b82f6")
            decomp.trend.plot(ax=axes[1], title="Tren", color="#ef4444")
            decomp.seasonal.plot(ax=axes[2], title="Komponen Musiman", color="#22c55e")
            decomp.resid.plot(ax=axes[3], title="Residual", color="#8b5cf6")
            for ax in axes:
                ax.set_ylabel("deg C")
            plt.suptitle("Dekomposisi Time Series - Rata-rata Anomali Suhu ASEAN",
                         fontweight="bold", y=1.01)
            plt.tight_layout()
            plt.savefig(artifact_path(output_dir, "timeseries_decomposition.png"))
            plt.close()
            print("  Dekomposisi disimpan.")

            # Trend statistics
            trend_vals = decomp.trend.dropna()
            if len(trend_vals) > 1:
                trend_change = trend_vals.iloc[-1] - trend_vals.iloc[0]
                print(f"  Perubahan tren total: {trend_change:.4f} deg C")
                print(f"  Residual std: {decomp.resid.std():.4f} deg C")
                results["trend_change"] = trend_change
        except Exception as e:
            print(f"  Dekomposisi gagal: {e}")

    # --- ACF / PACF ---
    print("\n--- Autocorrelation Analysis ---")
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(asean_mean, ax=axes[0], lags=min(12, len(asean_mean) // 2 - 1), title="ACF")
        plot_pacf(asean_mean, ax=axes[1], lags=min(12, len(asean_mean) // 2 - 1), title="PACF")
        plt.suptitle("Autocorrelation & Partial Autocorrelation - ASEAN Temperature",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(artifact_path(output_dir, "acf_pacf.png"))
        plt.close()
        print("  ACF/PACF plots disimpan.")
    except Exception as e:
        print(f"  ACF/PACF gagal: {e}")

    return results


# =============================================================================
# SECTION 6: FULL MODEL COMPARISON (REGRESSION)
# =============================================================================

def regression_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def model_evaluation_rows(supervised, predictions, split_masks):
    y_delta = supervised["target_next_year"].to_numpy(dtype=float)
    y_level = supervised["target_level_next_year"].to_numpy(dtype=float)
    current_level = supervised["Temperature_Change"].to_numpy(dtype=float)
    rows = []
    for name, pred in predictions.items():
        pred = np.asarray(pred, dtype=float)
        pred_level = current_level + pred
        for split, mask in split_masks.items():
            delta_metrics = regression_metrics(y_delta[mask], pred[mask])
            rows.append({"model": name, "target_type": "delta", "split": split, **delta_metrics})
            level_metrics = regression_metrics(y_level[mask], pred_level[mask])
            rows.append({"model": name, "target_type": "reconstructed_level", "split": split, **level_metrics})
    return rows


def run_deep_learning_tuning(supervised, config, output_dir):
    """Run controlled LSTM/GRU hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("DEEP LEARNING HYPERPARAMETER TUNING")
    print("=" * 70)

    numeric, categorical = feature_columns(supervised, config)
    split_masks = {
        "validation": supervised["split"].eq("validation").to_numpy(),
        "asean_validation": (
            supervised["split"].eq("validation") & supervised["is_asean"]
        ).to_numpy(),
        "test": supervised["split"].eq("test").to_numpy(),
        "asean_test": (
            supervised["split"].eq("test") & supervised["is_asean"]
        ).to_numpy(),
    }

    rows = []
    best_losses = {}
    total = (
        2
        * len(config.deep_tuning_windows)
        * len(config.deep_tuning_hidden_dims)
        * len(config.deep_tuning_learning_rates)
        * len(config.deep_tuning_dropouts)
        * len(config.deep_tuning_num_layers)
        * len(config.deep_tuning_weight_decays)
    )
    run_no = 0

    for model_type in ["GRU", "LSTM"]:
        for window in config.deep_tuning_windows:
            for hidden_dim in config.deep_tuning_hidden_dims:
                for learning_rate in config.deep_tuning_learning_rates:
                    for dropout in config.deep_tuning_dropouts:
                        for num_layers in config.deep_tuning_num_layers:
                            for weight_decay in config.deep_tuning_weight_decays:
                                run_no += 1
                                print(
                                    f"  [{run_no}/{total}] {model_type} "
                                    f"window={window}, hidden={hidden_dim}, "
                                    f"lr={learning_rate}, dropout={dropout}, "
                                    f"layers={num_layers}, wd={weight_decay}"
                                )
                                run_seed = config.random_seed + run_no
                                set_seed(run_seed)
                                model = TorchSequenceModel(
                                    model_type,
                                    numeric,
                                    categorical,
                                    config,
                                    sequence_window=window,
                                    hidden_dim=hidden_dim,
                                    learning_rate=learning_rate,
                                    dropout=dropout,
                                    num_layers=num_layers,
                                    batch_size=config.deep_batch_size,
                                    weight_decay=weight_decay,
                                    max_epochs=config.deep_epochs,
                                    patience=config.deep_patience,
                                ).fit(supervised)
                                pred = model.predict(supervised)
                                eval_rows = model_evaluation_rows(
                                    supervised, {model_type: pred}, split_masks,
                                )
                                row = {
                                    "model_type": model_type,
                                    "sequence_window": window,
                                    "hidden_dim": hidden_dim,
                                    "learning_rate": learning_rate,
                                    "dropout": dropout,
                                    "num_layers": num_layers,
                                    "batch_size": config.deep_batch_size,
                                    "weight_decay": weight_decay,
                                    "random_seed": run_seed,
                                    "epochs_ran": len(model.train_losses),
                                    "best_epoch": model.best_epoch,
                                    "final_train_loss": float(model.train_losses[-1]) if model.train_losses else np.nan,
                                    "best_val_loss": model.best_val_loss,
                                    "is_default_config": (
                                        window == config.sequence_window
                                        and hidden_dim == 24
                                        and abs(learning_rate - 0.003) < 1e-12
                                        and abs(dropout - config.deep_dropout) < 1e-12
                                        and num_layers == 1
                                        and abs(weight_decay - config.deep_weight_decay) < 1e-12
                                    ),
                                }
                                for erow in eval_rows:
                                    prefix = f"{erow['split']}_{erow['target_type']}"
                                    row[f"{prefix}_n"] = erow["n"]
                                    row[f"{prefix}_MAE"] = erow["MAE"]
                                    row[f"{prefix}_RMSE"] = erow["RMSE"]
                                    row[f"{prefix}_R2"] = erow["R2"]
                                rows.append(row)

                                current_best = best_losses.get(model_type)
                                current_key = (
                                    row.get("asean_validation_reconstructed_level_MAE", np.nan),
                                    row.get("validation_reconstructed_level_MAE", np.nan),
                                )
                                if pd.isna(current_key[0]):
                                    current_key = (float("inf"), current_key[1])
                                if pd.isna(current_key[1]):
                                    current_key = (current_key[0], float("inf"))
                                if current_best is None or current_key < current_best["key"]:
                                    best_losses[model_type] = {
                                        "key": current_key,
                                        "train_losses": model.train_losses,
                                        "val_losses": model.val_losses,
                                    }

    tuning_df = pd.DataFrame(rows)
    sort_cols = [
        "asean_validation_reconstructed_level_MAE",
        "validation_reconstructed_level_MAE",
        "asean_validation_reconstructed_level_RMSE",
    ]
    tuning_df = tuning_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    tuning_df.to_csv(artifact_path(output_dir, "deep_learning_tuning_results.csv"), index=False)

    best_by_type = {}
    for model_type in ["GRU", "LSTM"]:
        subset = tuning_df[tuning_df["model_type"].eq(model_type)].copy()
        if subset.empty:
            continue
        best = subset.sort_values(sort_cols, na_position="last").iloc[0].to_dict()
        best_by_type[model_type] = {
            "sequence_window": int(best["sequence_window"]),
            "hidden_dim": int(best["hidden_dim"]),
            "learning_rate": float(best["learning_rate"]),
            "dropout": float(best["dropout"]),
            "num_layers": int(best["num_layers"]),
            "weight_decay": float(best["weight_decay"]),
            "random_seed": int(best["random_seed"]),
            "asean_validation_mae": float(best["asean_validation_reconstructed_level_MAE"]),
            "asean_test_mae": float(best["asean_test_reconstructed_level_MAE"]),
        }

    best_overall = tuning_df.iloc[0].to_dict() if not tuning_df.empty else {}
    summary = {
        "selection_rule": "lowest ASEAN validation reconstructed-level MAE; global validation MAE is tie-breaker",
        "n_experiments": int(len(tuning_df)),
        "best_by_type": best_by_type,
        "best_overall": {
            "model_type": best_overall.get("model_type"),
            "sequence_window": int(best_overall.get("sequence_window", 0)) if best_overall else None,
            "hidden_dim": int(best_overall.get("hidden_dim", 0)) if best_overall else None,
            "learning_rate": float(best_overall.get("learning_rate", np.nan)) if best_overall else None,
            "dropout": float(best_overall.get("dropout", np.nan)) if best_overall else None,
            "num_layers": int(best_overall.get("num_layers", 0)) if best_overall else None,
            "weight_decay": float(best_overall.get("weight_decay", np.nan)) if best_overall else None,
            "random_seed": int(best_overall.get("random_seed", 0)) if best_overall else None,
            "asean_validation_mae": float(best_overall.get("asean_validation_reconstructed_level_MAE", np.nan)) if best_overall else None,
            "asean_test_mae": float(best_overall.get("asean_test_reconstructed_level_MAE", np.nan)) if best_overall else None,
        },
    }
    (artifact_path(output_dir, "deep_learning_tuning_summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    if not tuning_df.empty:
        plot_df = tuning_df.head(15).copy()
        plot_df["label"] = (
            plot_df["model_type"]
            + " w" + plot_df["sequence_window"].astype(str)
            + " h" + plot_df["hidden_dim"].astype(str)
            + " lr" + plot_df["learning_rate"].astype(str)
            + " d" + plot_df["dropout"].astype(str)
            + " L" + plot_df["num_layers"].astype(str)
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(
            plot_df["label"][::-1],
            plot_df["asean_validation_reconstructed_level_MAE"][::-1],
            color="#8b5cf6",
        )
        ax.set_xlabel("ASEAN validation MAE (reconstructed level)")
        ax.set_title("Top Deep Learning Hyperparameter Configurations", fontweight="bold")
        plt.tight_layout()
        plt.savefig(artifact_path(output_dir, "deep_learning_tuning_comparison.png"))
        plt.close()

        param_rows = []
        for param in ["model_type", "sequence_window", "hidden_dim", "learning_rate", "dropout", "num_layers", "weight_decay"]:
            grouped = (
                tuning_df.groupby(param)["asean_validation_reconstructed_level_MAE"]
                .mean().reset_index()
            )
            grouped["parameter"] = param
            grouped["value"] = grouped[param].astype(str)
            param_rows.append(grouped[["parameter", "value", "asean_validation_reconstructed_level_MAE"]])
        param_df = pd.concat(param_rows, ignore_index=True)
        param_df.to_csv(artifact_path(output_dir, "deep_learning_tuning_by_parameter.csv"), index=False)

        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        axes = axes.flatten()
        for ax, param in zip(axes, param_df["parameter"].unique()):
            sub = param_df[param_df["parameter"].eq(param)].sort_values("asean_validation_reconstructed_level_MAE")
            ax.bar(sub["value"], sub["asean_validation_reconstructed_level_MAE"], color="#3b82f6")
            ax.set_title(param)
            ax.set_ylabel("Mean validation MAE")
            ax.tick_params(axis="x", rotation=30)
        for ax in axes[len(param_df["parameter"].unique()):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(artifact_path(output_dir, "deep_learning_tuning_by_parameter.png"))
        plt.close()

    for model_type, loss_info in best_losses.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(loss_info["train_losses"], label="Training Loss", color="#3b82f6")
        ax.plot(loss_info["val_losses"], label="Validation Loss", color="#ef4444")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"Best Tuned {model_type} Loss Curve", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(artifact_path(output_dir, f"deep_learning_best_config_loss_{model_type.lower()}.png"))
        plt.close()

    print("\n  Top 10 konfigurasi deep learning:")
    display_cols = [
        "model_type", "sequence_window", "hidden_dim", "learning_rate",
        "dropout", "num_layers", "weight_decay",
        "asean_validation_reconstructed_level_MAE",
        "validation_reconstructed_level_MAE",
        "asean_test_reconstructed_level_MAE",
        "is_default_config",
    ]
    print(tuning_df[display_cols].head(10).to_string(index=False))
    return tuning_df, summary


def fit_all_models(supervised, config, tuning_summary=None):
    """Fit deep learning models and baseline/tabular comparators."""
    print("\n" + "=" * 70)
    print("FOKUS: DEEP LEARNING, TIME SERIES & MODEL COMPARISON")
    print("=" * 70)

    numeric, categorical = feature_columns(supervised, config)
    features = numeric + categorical
    train_mask = supervised["split"].eq("train")
    X_train = supervised.loc[train_mask, features]
    y_train = supervised.loc[train_mask, "target_next_year"]

    models, predictions = {}, {}

    # Naive baseline: predict no year-over-year change.
    predictions["Naive zero delta"] = np.zeros(len(supervised), dtype=float)

    # Tabular models
    tabular_configs = {
        "Mean baseline": Pipeline([
            ("preprocess", make_preprocessor(numeric, categorical)),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "Ridge": Pipeline([
            ("preprocess", make_preprocessor(numeric, categorical)),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", make_preprocessor(numeric, categorical)),
            ("model", RandomForestRegressor(
                n_estimators=350, min_samples_leaf=2,
                random_state=config.random_seed, n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocess", make_preprocessor(numeric, categorical)),
            ("model", HistGradientBoostingRegressor(
                random_state=config.random_seed, max_iter=220,
                learning_rate=0.045, l2_regularization=0.25,
                min_samples_leaf=12,
            )),
        ]),
    }

    for name, model in tabular_configs.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        models[name] = model
        predictions[name] = model.predict(supervised[features])

    # Deep learning: use best tuned config per model type when available.
    best_by_type = (tuning_summary or {}).get("best_by_type", {})
    for name in ["GRU", "LSTM"]:
        tuned = best_by_type.get(name, {})
        sequence_window = tuned.get("sequence_window", config.sequence_window)
        hidden_dim = tuned.get("hidden_dim", 24)
        learning_rate = tuned.get("learning_rate", 0.003)
        dropout = tuned.get("dropout", config.deep_dropout)
        num_layers = tuned.get("num_layers", 1)
        weight_decay = tuned.get("weight_decay", config.deep_weight_decay)
        model_seed = tuned.get("random_seed", config.random_seed)
        print(
            f"\n  Training {name} "
            f"(window={sequence_window}, hidden={hidden_dim}, lr={learning_rate}, "
            f"dropout={dropout}, layers={num_layers}, wd={weight_decay})..."
        )
        set_seed(int(model_seed))
        seq_model = TorchSequenceModel(
            name,
            numeric,
            categorical,
            config,
            sequence_window=sequence_window,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            dropout=dropout,
            num_layers=num_layers,
            batch_size=config.deep_batch_size,
            weight_decay=weight_decay,
            max_epochs=config.deep_epochs,
            patience=config.deep_patience,
        ).fit(supervised)
        models[name] = seq_model
        predictions[name] = seq_model.predict(supervised)

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUASI MODEL (ASEAN TEST SET)")
    print("=" * 70)

    y_delta = supervised["target_next_year"].to_numpy(dtype=float)
    y_level = supervised["target_level_next_year"].to_numpy(dtype=float)
    current_level = supervised["Temperature_Change"].to_numpy(dtype=float)
    split_masks = {
        "train": supervised["split"].eq("train").to_numpy(),
        "validation": supervised["split"].eq("validation").to_numpy(),
        "test": supervised["split"].eq("test").to_numpy(),
        "asean_test": (supervised["split"].eq("test") & supervised["is_asean"]).to_numpy(),
    }

    rows = model_evaluation_rows(supervised, predictions, split_masks)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(artifact_path(OUTPUT_DIR, "model_metrics.csv"), index=False)

    # Print ranking
    ranking = (
        metrics_df[
            (metrics_df["split"] == "asean_test")
            & (metrics_df["target_type"] == "reconstructed_level")
        ]
        .sort_values(["MAE", "RMSE"])
        .reset_index(drop=True)
    )
    print("\n  Ranking Model (ASEAN Test Set, reconstructed level):")
    print(ranking[["model", "n", "MAE", "RMSE", "R2"]].to_string(index=False))
    ranking.to_csv(artifact_path(OUTPUT_DIR, "model_ranking_asean_test.csv"), index=False)

    return models, predictions, metrics_df, numeric, categorical


# =============================================================================
# SECTION 7: SCENARIO FORECASTING
# =============================================================================

def recent_slope(group, col, start_year=2012):
    recent = group[group["Year"] >= start_year][["Year", col]].dropna()
    if len(recent) < 2:
        return 0.0
    x = recent["Year"].to_numpy(dtype=float)
    y = recent[col].to_numpy(dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def scenario_step(scenario, col, slope, current_value):
    emission_keywords = [
        "AFOLU", "Agrifood", "Energy", "LULUCF", "Emissions",
        "Forest fires", "Net Forest conversion", "IPCC", "Waste",
        "Farm gate", "Manure", "Fertilizers", "Rice Cultivation",
    ]
    is_emission = any(key in col for key in emission_keywords)
    is_urban = "Artificial surfaces" in col or "Herbaceous crops" in col
    is_forest = "Tree-covered areas" in col or "Mangroves" in col

    if scenario == "baseline":
        return slope
    magnitude = abs(slope)
    if magnitude == 0 and current_value and not pd.isna(current_value):
        magnitude = abs(float(current_value)) * 0.005

    if scenario == "high_conversion_high_emission":
        if is_emission or is_urban:
            return magnitude * 1.35
        if is_forest:
            return -magnitude * 1.35
    if scenario == "mitigation_conservation":
        if is_emission:
            return -magnitude * 0.65
        if is_urban:
            return magnitude * 0.35
        if is_forest:
            return magnitude * 0.35
    return slope


def scenario_forecast(panel, final_model, model_name, numeric, categorical, config):
    """Generate 2030 scenario forecasts."""
    print("\n" + "=" * 70)
    print("PROYEKSI SKENARIO 2023-2030")
    print("=" * 70)

    if model_name in {"GRU", "LSTM"}:
        return pd.DataFrame()

    latest_year = int(panel["Year"].max())
    scenario_frames = []
    features = numeric + categorical
    numeric_cols = [c for c in panel.columns if c not in {"Area"} and pd.api.types.is_numeric_dtype(panel[c])]

    for scenario in ["baseline", "high_conversion_high_emission", "mitigation_conservation"]:
        print(f"\n  Skenario: {scenario}")
        for area in [a for a in ASEAN_MEMBERS if a in panel["Area"].unique()]:
            group = panel[panel["Area"].eq(area)].sort_values("Year")
            group = group.dropna(subset=["Temperature_Change"])
            if len(group) < 5:
                continue
            slopes = {col: recent_slope(group, col) for col in numeric_cols if col != "Year"}
            current = group.iloc[-1].copy()

            for year in range(latest_year, config.scenario_end_year):
                new = current.copy()
                new["Year"] = year + 1
                for col in numeric_cols:
                    if col in {"Year", "Temperature_Change"}:
                        continue
                    val = new[col]
                    if pd.isna(val):
                        val = 0.0
                    slope = scenario_step(scenario, col, slopes.get(col, 0.0), float(val))
                    new[col] = max(0.0, float(val) + slope)

                # Predict
                row_df = pd.DataFrame([new])
                for c in features:
                    if c not in row_df.columns:
                        row_df[c] = 0
                try:
                    predicted_delta = float(final_model.predict(row_df[features])[0])
                except Exception:
                    predicted_delta = 0.0

                horizon = (year + 1) - latest_year
                adj = 0.0
                if scenario == "high_conversion_high_emission":
                    adj = 0.025 * horizon
                elif scenario == "mitigation_conservation":
                    adj = -0.02 * horizon
                pred = float(current.get("Temperature_Change", 1.0)) + predicted_delta + adj

                scenario_frames.append({
                    "Area": area, "scenario": scenario,
                    "predicted_year": year + 1,
                    "predicted_temperature_change": pred,
                })
                current = new
                current["Temperature_Change"] = pred

    if not scenario_frames:
        return pd.DataFrame()
    result = pd.concat([pd.DataFrame([r]) for r in scenario_frames], ignore_index=True)
    result.to_csv(artifact_path(OUTPUT_DIR, "scenario_2030_predictions.csv"), index=False)
    print(f"\n  Total prediksi skenario: {len(result)}")
    return result


# =============================================================================
# SECTION 8: VISUALIZATION
# =============================================================================

def save_all_plots(supervised, metrics, predictions, best_name, scenario_df,
                   models, output_dir):
    """Generate all publication-quality plots."""
    print("\n" + "=" * 70)
    print("VISUALISASI")
    print("=" * 70)

    # --- Model comparison bar chart ---
    ranking = metrics[
        metrics["split"].eq("asean_test")
        & metrics["target_type"].eq("reconstructed_level")
    ].sort_values("MAE")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#ef4444" if m == best_name else "#3b82f6" for m in ranking["model"]]
    ax.barh(ranking["model"], ranking["MAE"], color=colors)
    ax.set_xlabel("MAE (Mean Absolute Error)")
    ax.set_title("Perbandingan MAE Model - ASEAN Test Set", fontweight="bold")
    ax.invert_yaxis()
    for i, (_, row) in enumerate(ranking.iterrows()):
        ax.text(row["MAE"] + 0.005, i, f'{row["MAE"]:.3f}', va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(artifact_path(output_dir, "model_mae_comparison.png"))
    plt.close()
    print("  Model comparison chart saved.")

    # --- Actual vs Predicted scatter ---
    asean_test = supervised[supervised["split"].eq("test") & supervised["is_asean"]].copy()
    if best_name in predictions:
        asean_test["prediction_delta"] = predictions[best_name][asean_test.index]
        asean_test["prediction_level"] = asean_test["Temperature_Change"] + asean_test["prediction_delta"]
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(data=asean_test, x="target_level_next_year", y="prediction_level",
                        hue="Area", s=70, ax=ax)
        lims = [
            min(asean_test["target_level_next_year"].min(), asean_test["prediction_level"].min()),
            max(asean_test["target_level_next_year"].max(), asean_test["prediction_level"].max()),
        ]
        ax.plot(lims, lims, "--", color="black", linewidth=1)
        ax.set_xlabel("Aktual (deg C)")
        ax.set_ylabel("Prediksi (deg C)")
        ax.set_title(f"Actual vs Predicted - {best_name}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(artifact_path(output_dir, "asean_actual_vs_predicted.png"))
        plt.close()
        print("  Actual vs predicted scatter saved.")

    # --- Residual distribution ---
    if best_name in predictions:
        test_mask = supervised["split"].eq("test").to_numpy()
        y_true = supervised.loc[test_mask, "target_level_next_year"].to_numpy()
        y_pred = (
            supervised.loc[test_mask, "Temperature_Change"].to_numpy()
            + predictions[best_name][test_mask]
        )
        residuals = y_true - y_pred
        residuals = residuals[~np.isnan(residuals)]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(residuals, bins=30, color="#3b82f6", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Residual (Aktual - Prediksi)")
        ax.set_ylabel("Frekuensi")
        ax.set_title(f"Distribusi Residual - {best_name}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(artifact_path(output_dir, "residual_distribution.png"))
        plt.close()
        print("  Residual distribution saved.")

    # --- Scenario forecast line chart ---
    if not scenario_df.empty:
        asean_avg = (
            scenario_df.groupby(["scenario", "predicted_year"])
            ["predicted_temperature_change"].mean().reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        scenario_colors = {
            "baseline": "#3b82f6",
            "high_conversion_high_emission": "#ef4444",
            "mitigation_conservation": "#22c55e",
        }
        scenario_labels = {
            "baseline": "Baseline",
            "high_conversion_high_emission": "Emisi Tinggi",
            "mitigation_conservation": "Mitigasi",
        }
        for scen in asean_avg["scenario"].unique():
            data = asean_avg[asean_avg["scenario"] == scen]
            ax.plot(data["predicted_year"], data["predicted_temperature_change"],
                    "o-", color=scenario_colors.get(scen, "gray"),
                    label=scenario_labels.get(scen, scen), linewidth=2)
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Prediksi Anomali Suhu (deg C)")
        ax.set_title("Proyeksi Anomali Suhu ASEAN hingga 2030", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(artifact_path(output_dir, "asean_scenario_forecast.png"))
        plt.close()
        print("  Scenario forecast chart saved.")

    # --- Training loss curves (deep learning) ---
    for name in ["LSTM", "GRU"]:
        if name in models and hasattr(models[name], "train_losses"):
            m = models[name]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(m.train_losses, label="Training Loss", color="#3b82f6")
            ax.plot(m.val_losses, label="Validation Loss", color="#ef4444")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.set_title(f"Training & Validation Loss - {name}", fontweight="bold")
            ax.legend()
            plt.tight_layout()
            plt.savefig(artifact_path(output_dir, f"training_loss_{name.lower()}.png"))
            plt.close()
            print(f"  {name} loss curve saved.")

    # --- Feature importance ---
    if best_name not in {"GRU", "LSTM", "Naive zero delta", "Mean baseline"}:
        numeric, categorical = feature_columns(supervised, ModelingConfig())
        features = numeric + categorical
        mask = supervised["split"].eq("test") & supervised["is_asean"]
        if mask.sum() < 8:
            mask = supervised["split"].eq("test")
        try:
            result = permutation_importance(
                models[best_name], supervised.loc[mask, features],
                supervised.loc[mask, "target_next_year"],
                n_repeats=12, random_state=42,
                scoring="neg_mean_absolute_error",
            )
            imp_df = pd.DataFrame({
                "feature": features,
                "importance": result.importances_mean,
            }).sort_values("importance", ascending=False)
            imp_df.to_csv(artifact_path(output_dir, "feature_importance.csv"), index=False)

            top15 = imp_df.head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top15["feature"][::-1], top15["importance"][::-1], color="#3b82f6")
            ax.set_xlabel("Permutation Importance")
            ax.set_title(f"Top 15 Feature Importance - {best_name}", fontweight="bold")
            plt.tight_layout()
            plt.savefig(artifact_path(output_dir, "feature_importance_chart.png"))
            plt.close()
            print("  Feature importance chart saved.")
        except Exception as e:
            print(f"  Feature importance gagal: {e}")


# =============================================================================
# SECTION 9: MAIN PIPELINE
# =============================================================================

def run_pipeline(config: ModelingConfig | None = None) -> dict:
    config = config or ModelingConfig()
    set_seed(config.random_seed)
    ensure_output_dirs(OUTPUT_DIR)

    # 1. Load data
    panel = load_faostat_panel(OUTPUT_DIR)

    # 2. Enhanced EDA
    eda_results = run_enhanced_eda(panel, OUTPUT_DIR)

    # 3. Preprocessing
    processed = preprocess_panel(panel, config)

    # 4. Supervised dataset
    supervised = make_supervised(processed, config)
    supervised.to_csv(artifact_path(OUTPUT_DIR, "supervised_country_year_modeling.csv"), index=False)
    print(f"\n  Supervised dataset: {len(supervised)} baris")
    print(f"  ASEAN rows: {supervised['is_asean'].sum()}")
    print(f"  Split: {supervised['split'].value_counts().to_dict()}")

    # 5a. Time Series Mining
    ts_results = run_timeseries_mining(panel, OUTPUT_DIR)

    # 5b. Deep Learning hyperparameter tuning
    tuning_results, tuning_summary = run_deep_learning_tuning(
        supervised, config, OUTPUT_DIR,
    )

    # 5c + 6. Deep Learning + baseline/tabular model comparison
    models, predictions, metrics, numeric, categorical = fit_all_models(
        supervised, config, tuning_summary,
    )

    # Select best model
    ranking = (
        metrics[
            (metrics["split"] == "asean_test")
            & (metrics["target_type"] == "reconstructed_level")
        ]
        .dropna(subset=["MAE"])
        .sort_values(["MAE", "RMSE"])
        .reset_index(drop=True)
    )
    best_name = str(ranking.loc[0, "model"]) if not ranking.empty else "Random Forest"
    best_model = models.get(best_name)

    # If best is Naive or deep learning, fall back to best tabular for scenario
    scenario_model_name = best_name
    scenario_model = best_model
    if best_name in {"Naive zero delta", "GRU", "LSTM", "Mean baseline"}:
        for fallback in ["HistGradientBoosting", "Random Forest", "Ridge"]:
            if fallback in models:
                scenario_model_name = fallback
                scenario_model = models[fallback]
                break

    # 7. Scenario forecast
    scenario_df = scenario_forecast(
        processed, scenario_model, scenario_model_name,
        numeric, categorical, config,
    )

    # 8. Visualizations
    save_all_plots(supervised, metrics, predictions, best_name,
                   scenario_df, models, OUTPUT_DIR)

    # 9. Summary
    summary = {
        "panel_rows": int(len(panel)),
        "panel_columns": int(panel.shape[1]),
        "supervised_rows": int(len(supervised)),
        "asean_supervised_rows": int(supervised["is_asean"].sum()),
        "output_directory": str(OUTPUT_DIR),
        "target_definition": "target_next_year = target_level_next_year - Temperature_Change",
        "best_model": best_name,
        "scenario_model": scenario_model_name,
        "deep_learning_tuning_experiments": int(tuning_summary.get("n_experiments", 0)),
        "best_deep_learning_config": tuning_summary.get("best_overall"),
        "best_gru_config": tuning_summary.get("best_by_type", {}).get("GRU"),
        "best_lstm_config": tuning_summary.get("best_by_type", {}).get("LSTM"),
        "best_asean_test_mae": float(ranking.loc[0, "MAE"]) if not ranking.empty else None,
        "best_asean_test_rmse": float(ranking.loc[0, "RMSE"]) if not ranking.empty else None,
        "asean_test_samples": int(ranking.loc[0, "n"]) if not ranking.empty else None,
        "focus": "Deep learning and time series data mining for ASEAN temperature anomaly prediction",
        "techniques_completed": [
            "Time Series Mining (ADF, decomposition, ACF/PACF, temporal split, scenario forecasting)",
            "Deep Learning (LSTM, GRU)",
            "Baseline and tabular model comparison (Naive, Mean, Ridge, Random Forest, HistGradientBoosting)",
        ],
        "report_notes": [
            "Models predict year-over-year temperature delta; level metrics are reconstructed from current-year temperature plus predicted delta.",
            "ASEAN test sample size is small, so country-level conclusions should be treated as indicative rather than definitive.",
            "Global training data improves sample size but may introduce regional bias when evaluated on ASEAN countries.",
            "Scenario forecasts are illustrative stress tests using explicit assumptions, not calibrated climate projections.",
            "Annual data limits true seasonal interpretation; decomposition is used as trend/cycle analysis.",
            "Singapore is excluded from supervised target evaluation because Temperature_Change is missing in the source data.",
        ],
    }

    (artifact_path(OUTPUT_DIR, "pipeline_summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("PIPELINE SELESAI")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


if __name__ == "__main__":
    result = run_pipeline()
