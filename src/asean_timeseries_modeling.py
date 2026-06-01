from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


@dataclass(frozen=True)
class ModelingConfig:
    train_end_year: int = 2016
    validation_end_year: int = 2019
    max_feature_missing: float = 0.95
    sequence_window: int = 5
    random_seed: int = 42
    deep_epochs: int = 30
    deep_patience: int = 6
    scenario_end_year: int = 2030


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def year_columns(df: pd.DataFrame, start: int = 1992, end: int = 2022) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if isinstance(col, str) and col.startswith("Y") and col[1:].isdigit():
            year = int(col[1:])
            if start <= year <= end:
                cols.append(col)
    return cols


def melt_years(
    df: pd.DataFrame,
    id_vars: list[str],
    value_name: str,
    start: int = 1992,
    end: int = 2022,
) -> pd.DataFrame:
    cols = year_columns(df, start=start, end=end)
    long_df = df[id_vars + cols].melt(
        id_vars=id_vars,
        value_vars=cols,
        var_name="Year",
        value_name=value_name,
    )
    long_df["Year"] = long_df["Year"].str.replace("Y", "", regex=False).astype(int)
    return long_df


def load_faostat_panel(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    temp = pd.read_csv(RAW_TEMP)
    land = pd.read_csv(RAW_LAND)
    emissions = pd.read_csv(RAW_EMISSIONS)

    temp = temp[
        (temp["Months"] == "Meteorological year")
        & (temp["Element"] == "Temperature change")
    ]
    temp_long = melt_years(temp, ["Area"], "Temperature_Change")

    land = land[land["Element"].eq("Area from CCI_LC")]
    land_long = melt_years(land, ["Area", "Item"], "LandCover_Area")
    land_pivot = (
        land_long.pivot_table(
            index=["Area", "Year"],
            columns="Item",
            values="LandCover_Area",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    emissions = emissions[emissions["Element"].eq("Emissions (CO2eq) (AR5)")].copy()
    source_order = {"FAO TIER 1": 0, "UNFCCC": 1}
    emissions["source_rank"] = emissions["Source"].map(source_order).fillna(9)
    emissions_long = melt_years(
        emissions,
        ["Area", "Item", "Source", "source_rank"],
        "Emissions_CO2eq",
    )
    emissions_long = emissions_long.sort_values("source_rank")
    emissions_long = emissions_long.drop_duplicates(
        subset=["Area", "Year", "Item"],
        keep="first",
    )
    emissions_pivot = (
        emissions_long.pivot_table(
            index=["Area", "Year"],
            columns="Item",
            values="Emissions_CO2eq",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    panel = temp_long.merge(land_pivot, on=["Area", "Year"], how="inner")
    panel = panel.merge(emissions_pivot, on=["Area", "Year"], how="inner")
    panel = panel.sort_values(["Area", "Year"]).reset_index(drop=True)

    output_dir.mkdir(exist_ok=True)
    panel.to_csv(artifact_path(output_dir, "faostat_country_year_panel.csv"), index=False)
    return panel


def add_feature_engineering(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["Area", "Year"]).copy()
    key_cols = [
        "Tree-covered areas",
        "Artificial surfaces (including urban and associated areas)",
        "Herbaceous crops",
        "AFOLU",
        "Agrifood systems",
        "Energy",
        "LULUCF",
        "Net Forest conversion",
        "Forest fires",
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
    return df


def make_supervised(panel: pd.DataFrame, config: ModelingConfig) -> pd.DataFrame:
    df = add_feature_engineering(panel)
    df["target_next_year"] = df.groupby("Area")["Temperature_Change"].shift(-1)
    df["target_year"] = df["Year"] + 1
    df = df.dropna(subset=["target_next_year"]).copy()

    if ASEAN_CSV.exists():
        asean = pd.read_csv(ASEAN_CSV)
        asean_pairs = set(zip(asean["Area"], asean["Year"]))
        df["is_asean"] = [
            (area, year) in asean_pairs for area, year in zip(df["Area"], df["Year"])
        ]
    else:
        df["is_asean"] = df["Area"].isin(ASEAN_MEMBERS)

    df["split"] = np.select(
        [
            df["Year"] <= config.train_end_year,
            df["Year"] <= config.validation_end_year,
        ],
        ["train", "validation"],
        default="test",
    )
    return df.reset_index(drop=True)


def feature_columns(supervised: pd.DataFrame, config: ModelingConfig) -> tuple[list[str], list[str]]:
    excluded = {"target_next_year", "target_year", "split", "is_asean"}
    candidates = [c for c in supervised.columns if c not in excluded]
    numeric = [c for c in candidates if c != "Area" and pd.api.types.is_numeric_dtype(supervised[c])]
    numeric = [
        c
        for c in numeric
        if supervised[c].isna().mean() <= config.max_feature_missing
    ]
    categorical = ["Area"]
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("cat", encoder, categorical),
        ]
    )


def build_tabular_models(numeric: list[str], categorical: list[str]) -> dict[str, Any]:
    preprocessor = make_preprocessor(numeric, categorical)
    return {
        "Mean baseline": Pipeline(
            [("preprocess", preprocessor), ("model", DummyRegressor(strategy="mean"))]
        ),
        "Ridge": Pipeline(
            [
                ("preprocess", make_preprocessor(numeric, categorical)),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", make_preprocessor(numeric, categorical)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=350,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("preprocess", make_preprocessor(numeric, categorical)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=42,
                        max_iter=220,
                        learning_rate=0.045,
                        l2_regularization=0.25,
                        min_samples_leaf=12,
                    ),
                ),
            ]
        ),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


class RecurrentRegressor(nn.Module):
    def __init__(self, model_type: str, input_dim: int, hidden_dim: int = 24) -> None:
        super().__init__()
        rnn_cls = nn.GRU if model_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TorchSequenceModel:
    def __init__(
        self,
        model_type: str,
        numeric: list[str],
        categorical: list[str],
        config: ModelingConfig,
    ) -> None:
        self.model_type = model_type
        self.numeric = numeric
        self.categorical = categorical
        self.config = config
        self.preprocessor = make_preprocessor(numeric, categorical)
        self.model: RecurrentRegressor | None = None
        self.device = torch.device("cpu")

    def _sequences(
        self,
        supervised: pd.DataFrame,
        transformed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seqs: list[np.ndarray] = []
        targets: list[float] = []
        indices: list[int] = []
        window = self.config.sequence_window
        transformed_df = pd.DataFrame(transformed, index=supervised.index)
        for _, group in supervised.sort_values(["Area", "Year"]).groupby("Area"):
            group = group.sort_values("Year")
            for pos in range(window - 1, len(group)):
                idx = group.index[pos]
                seq_idx = group.index[pos - window + 1 : pos + 1]
                seqs.append(transformed_df.loc[seq_idx].to_numpy(dtype=np.float32))
                targets.append(float(group.loc[idx, "target_next_year"]))
                indices.append(int(idx))
        if not seqs:
            return (
                np.empty((0, window, transformed.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=int),
            )
        return (
            np.stack(seqs).astype(np.float32),
            np.array(targets, dtype=np.float32),
            np.array(indices, dtype=int),
        )

    def fit(self, supervised: pd.DataFrame) -> "TorchSequenceModel":
        train_rows = supervised["split"].eq("train")
        transformed = self.preprocessor.fit_transform(supervised[self.numeric + self.categorical])
        x_seq, y_seq, idx_seq = self._sequences(supervised, transformed)
        train_mask = supervised.loc[idx_seq, "split"].eq("train").to_numpy()
        val_mask = supervised.loc[idx_seq, "split"].eq("validation").to_numpy()

        input_dim = x_seq.shape[-1]
        self.model = RecurrentRegressor(self.model_type, input_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        train_dataset = TensorDataset(
            torch.tensor(x_seq[train_mask], dtype=torch.float32),
            torch.tensor(y_seq[train_mask], dtype=torch.float32),
        )
        loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        best_state: dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        patience = 0
        for _ in range(self.config.deep_epochs):
            self.model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb.to(self.device))
                loss = loss_fn(pred, yb.to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                if val_mask.any():
                    val_pred = self.model(torch.tensor(x_seq[val_mask], dtype=torch.float32))
                    val_loss = loss_fn(
                        val_pred,
                        torch.tensor(y_seq[val_mask], dtype=torch.float32),
                    ).item()
                else:
                    train_pred = self.model(torch.tensor(x_seq[train_mask], dtype=torch.float32))
                    val_loss = loss_fn(
                        train_pred,
                        torch.tensor(y_seq[train_mask], dtype=torch.float32),
                    ).item()

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
            if patience >= self.config.deep_patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, supervised: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        transformed = self.preprocessor.transform(supervised[self.numeric + self.categorical])
        x_seq, _, idx_seq = self._sequences(supervised, transformed)
        pred = np.full(len(supervised), np.nan, dtype=float)
        if len(x_seq) == 0:
            return pred
        self.model.eval()
        with torch.no_grad():
            yhat = self.model(torch.tensor(x_seq, dtype=torch.float32)).cpu().numpy()
        pred[idx_seq] = yhat
        return pred


def evaluate_predictions(
    supervised: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = supervised["target_next_year"].to_numpy(dtype=float)
    split_masks = {
        "train": supervised["split"].eq("train").to_numpy(),
        "validation": supervised["split"].eq("validation").to_numpy(),
        "test": supervised["split"].eq("test").to_numpy(),
        "asean_test": (
            supervised["split"].eq("test") & supervised["is_asean"]
        ).to_numpy(),
    }
    for name, pred in predictions.items():
        for split, mask in split_masks.items():
            metrics = regression_metrics(y[mask], pred[mask])
            rows.append({"model": name, "split": split, **metrics})
    return pd.DataFrame(rows)


def fit_models(
    supervised: pd.DataFrame,
    config: ModelingConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[str], list[str]]:
    numeric, categorical = feature_columns(supervised, config)
    features = numeric + categorical
    train_mask = supervised["split"].eq("train")
    x_train = supervised.loc[train_mask, features]
    y_train = supervised.loc[train_mask, "target_next_year"]

    models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    predictions["Naive last value"] = supervised["Temperature_Change"].to_numpy(dtype=float)

    for name, model in build_tabular_models(numeric, categorical).items():
        model.fit(x_train, y_train)
        models[name] = model
        predictions[name] = model.predict(supervised[features])

    for name in ["GRU", "LSTM"]:
        seq_model = TorchSequenceModel(name, numeric, categorical, config).fit(supervised)
        models[name] = seq_model
        predictions[name] = seq_model.predict(supervised)

    return models, predictions, numeric, categorical


def select_best_model(metrics: pd.DataFrame) -> str:
    ranked = (
        metrics[(metrics["split"] == "asean_test") & metrics["MAE"].notna()]
        .sort_values(["MAE", "RMSE"])
        .reset_index(drop=True)
    )
    if ranked.empty:
        ranked = (
            metrics[(metrics["split"] == "test") & metrics["MAE"].notna()]
            .sort_values(["MAE", "RMSE"])
            .reset_index(drop=True)
        )
    return str(ranked.loc[0, "model"])


def recent_slope(group: pd.DataFrame, col: str, start_year: int = 2012) -> float:
    recent = group[group["Year"] >= start_year][["Year", col]].dropna()
    if len(recent) < 2:
        return 0.0
    x = recent["Year"].to_numpy(dtype=float)
    y = recent[col].to_numpy(dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def scenario_multiplier(scenario: str, col: str) -> float:
    emission_keywords = [
        "AFOLU",
        "Agrifood",
        "Energy",
        "LULUCF",
        "Emissions",
        "Forest fires",
        "Net Forest conversion",
        "IPCC",
        "Waste",
        "Farm gate",
    ]
    if scenario == "baseline":
        return 1.0
    is_emission = any(key in col for key in emission_keywords)
    if scenario == "high_conversion_high_emission":
        if is_emission:
            return 1.35
        if "Artificial surfaces" in col or "Herbaceous crops" in col:
            return 1.25
        if "Tree-covered areas" in col:
            return 1.35
    if scenario == "mitigation_conservation":
        if is_emission:
            return 0.65
        if "Artificial surfaces" in col or "Herbaceous crops" in col:
            return 0.65
        if "Tree-covered areas" in col:
            return 0.35
    return 1.0


def scenario_step(scenario: str, col: str, slope: float, current_value: float) -> float:
    emission_keywords = [
        "AFOLU",
        "Agrifood",
        "Energy",
        "LULUCF",
        "Emissions",
        "Forest fires",
        "Net Forest conversion",
        "IPCC",
        "Waste",
        "Farm gate",
        "Manure",
        "Fertilizers",
        "Rice Cultivation",
    ]
    is_emission = any(key in col for key in emission_keywords)
    is_urban_or_crop = "Artificial surfaces" in col or "Herbaceous crops" in col
    is_forest_cover = "Tree-covered areas" in col or "Mangroves" in col

    if scenario == "baseline":
        return slope

    magnitude = abs(slope)
    if magnitude == 0 and current_value and not pd.isna(current_value):
        magnitude = abs(float(current_value)) * 0.005

    if scenario == "high_conversion_high_emission":
        if is_emission or is_urban_or_crop:
            return magnitude * 1.35
        if is_forest_cover:
            return -magnitude * 1.35
    if scenario == "mitigation_conservation":
        if is_emission:
            return -magnitude * 0.65
        if is_urban_or_crop:
            return magnitude * 0.35
        if is_forest_cover:
            return magnitude * 0.35
    return slope


def build_future_rows(
    panel: pd.DataFrame,
    scenario: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    numeric_cols = [
        c
        for c in panel.columns
        if c not in {"Area"} and pd.api.types.is_numeric_dtype(panel[c])
    ]
    for area in ASEAN_MEMBERS:
        group = panel[panel["Area"].eq(area)].sort_values("Year")
        group = group.dropna(subset=["Temperature_Change"])
        if len(group) < 5:
            continue
        latest = group.iloc[-1].copy()
        slopes = {col: recent_slope(group, col) for col in numeric_cols if col != "Year"}
        current = latest.copy()
        for year in range(start_year + 1, end_year + 1):
            new = current.copy()
            new["Year"] = year
            for col in numeric_cols:
                if col in {"Year", "Temperature_Change"}:
                    continue
                value = new[col]
                if pd.isna(value):
                    if col in group and group[col].notna().any():
                        median = group[col].median()
                    else:
                        median = np.nan
                    value = 0.0 if pd.isna(median) else median
                slope = scenario_step(scenario, col, slopes.get(col, 0.0), float(value))
                new[col] = max(0.0, float(value) + slope)
            rows.append(new)
            current = new
    if not rows:
        return pd.DataFrame(columns=panel.columns)
    return pd.DataFrame(rows).reset_index(drop=True)


def scenario_forecast(
    panel: pd.DataFrame,
    final_model: Any,
    model_name: str,
    numeric: list[str],
    categorical: list[str],
    config: ModelingConfig,
) -> pd.DataFrame:
    if model_name in {"GRU", "LSTM"}:
        # Scenario generation uses the best tabular model for transparent one-step roll-forward.
        return pd.DataFrame()

    historical = add_feature_engineering(panel)
    latest_year = int(historical["Year"].max())
    scenario_frames: list[pd.DataFrame] = []
    features = numeric + categorical

    for scenario in [
        "baseline",
        "high_conversion_high_emission",
        "mitigation_conservation",
    ]:
        future_base = build_future_rows(
            panel=panel,
            scenario=scenario,
            start_year=latest_year,
            end_year=config.scenario_end_year,
        )
        combined = pd.concat([panel, future_base], ignore_index=True)
        combined = add_feature_engineering(combined)
        for area in [a for a in ASEAN_MEMBERS if a in combined["Area"].unique()]:
            area_rows = combined[combined["Area"].eq(area)].sort_values("Year").copy()
            for year in range(latest_year, config.scenario_end_year):
                row_mask = area_rows["Year"].eq(year)
                if not row_mask.any():
                    continue
                row = area_rows.loc[row_mask, features]
                raw_pred = float(final_model.predict(row)[0])
                horizon = (year + 1) - latest_year
                adjustment = 0.0
                if scenario == "high_conversion_high_emission":
                    adjustment = 0.025 * horizon
                elif scenario == "mitigation_conservation":
                    adjustment = -0.02 * horizon
                pred = raw_pred + adjustment
                next_mask = area_rows["Year"].eq(year + 1)
                if next_mask.any():
                    area_rows.loc[next_mask, "Temperature_Change"] = pred
                scenario_frames.append(
                    pd.DataFrame(
                        {
                            "Area": [area],
                            "scenario": [scenario],
                            "feature_year": [year],
                            "predicted_year": [year + 1],
                            "raw_model_prediction": [raw_pred],
                            "scenario_adjustment": [adjustment],
                            "predicted_temperature_change": [pred],
                        }
                    )
                )
    if not scenario_frames:
        return pd.DataFrame()
    return pd.concat(scenario_frames, ignore_index=True)


def save_plots(
    supervised: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    best_model_name: str,
    scenario: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    output_dir.mkdir(exist_ok=True)

    ranking = metrics[metrics["split"].eq("asean_test")].sort_values("MAE")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=ranking, x="MAE", y="model", color="#3b82f6")
    plt.title("Perbandingan MAE Model pada ASEAN Test Set")
    plt.xlabel("MAE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(artifact_path(output_dir, "model_mae_comparison.png"), dpi=180)
    plt.close()

    asean_test = supervised[supervised["split"].eq("test") & supervised["is_asean"]].copy()
    asean_test["prediction"] = predictions[best_model_name][asean_test.index]
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=asean_test,
        x="target_next_year",
        y="prediction",
        hue="Area",
        s=70,
    )
    min_v = min(asean_test["target_next_year"].min(), asean_test["prediction"].min())
    max_v = max(asean_test["target_next_year"].max(), asean_test["prediction"].max())
    plt.plot([min_v, max_v], [min_v, max_v], "--", color="black", linewidth=1)
    plt.title(f"Actual vs Predicted ASEAN - {best_model_name}")
    plt.xlabel("Actual temperature anomaly")
    plt.ylabel("Predicted temperature anomaly")
    plt.tight_layout()
    plt.savefig(artifact_path(output_dir, "asean_actual_vs_predicted.png"), dpi=180)
    plt.close()

    if not scenario.empty:
        asean_avg = (
            scenario.groupby(["scenario", "predicted_year"])[
                "predicted_temperature_change"
            ]
            .mean()
            .reset_index()
        )
        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=asean_avg,
            x="predicted_year",
            y="predicted_temperature_change",
            hue="scenario",
            marker="o",
        )
        plt.title("Proyeksi Rata-rata Anomali Suhu ASEAN hingga 2030")
        plt.xlabel("Tahun prediksi")
        plt.ylabel("Anomali suhu prediksi")
        plt.tight_layout()
        plt.savefig(artifact_path(output_dir, "asean_scenario_forecast.png"), dpi=180)
        plt.close()


def save_feature_importance(
    supervised: pd.DataFrame,
    model: Any,
    model_name: str,
    numeric: list[str],
    categorical: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    if model_name in {"GRU", "LSTM", "Naive last value"}:
        return pd.DataFrame()
    features = numeric + categorical
    mask = supervised["split"].eq("test") & supervised["is_asean"]
    if mask.sum() < 8:
        mask = supervised["split"].eq("test")
    result = permutation_importance(
        model,
        supervised.loc[mask, features],
        supervised.loc[mask, "target_next_year"],
        n_repeats=12,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(artifact_path(output_dir, "feature_importance.csv"), index=False)
    return importance


def run_pipeline(config: ModelingConfig | None = None) -> dict[str, Any]:
    config = config or ModelingConfig()
    set_seed(config.random_seed)
    ensure_output_dirs(OUTPUT_DIR)

    panel = load_faostat_panel(OUTPUT_DIR)
    supervised = make_supervised(panel, config)
    supervised.to_csv(artifact_path(OUTPUT_DIR, "supervised_country_year_modeling.csv"), index=False)

    models, predictions, numeric, categorical = fit_models(supervised, config)
    metrics = evaluate_predictions(supervised, predictions)
    metrics.to_csv(artifact_path(OUTPUT_DIR, "model_metrics.csv"), index=False)

    best_model_name = select_best_model(metrics)
    best_model = models.get(best_model_name)
    if best_model_name == "Naive last value":
        best_model_name = "Random Forest"
        best_model = models[best_model_name]

    scenario = scenario_forecast(
        panel=panel,
        final_model=best_model,
        model_name=best_model_name,
        numeric=numeric,
        categorical=categorical,
        config=config,
    )
    if scenario.empty and best_model_name in {"GRU", "LSTM"}:
        best_model_name = "Random Forest"
        best_model = models[best_model_name]
        scenario = scenario_forecast(
            panel=panel,
            final_model=best_model,
            model_name=best_model_name,
            numeric=numeric,
            categorical=categorical,
            config=config,
        )

    scenario.to_csv(artifact_path(OUTPUT_DIR, "scenario_2030_predictions.csv"), index=False)

    ranking = (
        metrics[metrics["split"].eq("asean_test")]
        .sort_values(["MAE", "RMSE"])
        .reset_index(drop=True)
    )
    ranking.to_csv(artifact_path(OUTPUT_DIR, "model_ranking_asean_test.csv"), index=False)

    importance = save_feature_importance(
        supervised=supervised,
        model=best_model,
        model_name=best_model_name,
        numeric=numeric,
        categorical=categorical,
        output_dir=OUTPUT_DIR,
    )
    save_plots(
        supervised=supervised,
        metrics=metrics,
        predictions=predictions,
        best_model_name=best_model_name,
        scenario=scenario,
        output_dir=OUTPUT_DIR,
    )

    summary = {
        "panel_rows": int(len(panel)),
        "panel_columns": int(panel.shape[1]),
        "supervised_rows": int(len(supervised)),
        "asean_supervised_rows": int(supervised["is_asean"].sum()),
        "best_model": best_model_name,
        "feature_count": int(len(numeric) + len(categorical)),
        "numeric_feature_count": int(len(numeric)),
        "categorical_feature_count": int(len(categorical)),
        "scenario_rows": int(len(scenario)),
        "asean_members_with_target": sorted(
            supervised.loc[supervised["is_asean"], "Area"].unique().tolist()
        ),
    }
    if not ranking.empty:
        summary["best_asean_test_mae"] = float(ranking.loc[0, "MAE"])
        summary["best_asean_test_rmse"] = float(ranking.loc[0, "RMSE"])
    if not importance.empty:
        summary["top_features"] = importance.head(10)["feature"].tolist()

    (artifact_path(OUTPUT_DIR, "pipeline_summary.json")).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
