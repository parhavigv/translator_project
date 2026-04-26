"""
Adaptive Deep Learning Pipeline for Climate Data
Optimized version: vectorized drift detection, batched inference,
cached sequences, unified scaler inverse-transform, clean CLI.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── Optional TensorFlow ──────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available – using scikit-learn fallback.")

COLUMNS = ["YEAR", "MO", "DY", "T2M", "RH2M", "PRECTOTCORR",
           "WS2M", "PS", "T2M_MAX", "T2M_MIN", "WD10M", "ALLSKY_SFC_LW_DWN"]


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 – DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def resolve_data_path(filepath: str) -> str:
    """Find the dataset, checking common locations and name variants."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filepath),
        script_dir / filepath,
        *[script_dir / n for n in ("Minor project dataset.csv", "Minor project data.csv")],
        *[Path.cwd() / n for n in ("Minor project dataset.csv", "Minor project data.csv")],
    ]
    for p in dict.fromkeys(c.resolve() for c in candidates):   # preserve order, deduplicate
        if p.is_file():
            return str(p)
    checked = "\n".join(f"  - {p}" for p in dict.fromkeys(c.resolve() for c in candidates))
    raise FileNotFoundError(f"Dataset not found. Checked:\n{checked}")


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load NASA POWER CSV, impute missing values, build a DatetimeIndex."""
    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    # Try comma-separated first, then whitespace-separated
    try:
        df = pd.read_csv(filepath, usecols=COLUMNS)
        if not set(COLUMNS).issubset(df.columns):
            raise ValueError
    except Exception:
        df = pd.read_csv(
            filepath, sep=r"\s+", comment="#",
            names=COLUMNS, header=None, engine="python",
        )
        df = df[pd.to_numeric(df["YEAR"], errors="coerce").notna()].copy()

    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace(-999.0, np.nan, inplace=True)

    print(f"  Raw records loaded    : {len(df)}")
    print(f"  Missing values before : {df.isnull().sum().sum()}")

    # Impute: forward-fill → backward-fill → column mean
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("No valid rows after preprocessing. Check DATA_PATH and file format.")

    print(f"  Missing values after  : {df.isnull().sum().sum()}")

    df["DATE"] = pd.to_datetime(
        df[["YEAR", "MO", "DY"]].rename(columns={"YEAR": "year", "MO": "month", "DY": "day"})
    )
    df.set_index("DATE", inplace=True)
    df.drop(columns=["YEAR", "MO", "DY"], inplace=True)
    df.sort_index(inplace=True)

    print(f"  Date range : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Total days : {len(df)}")
    return df


def normalize_data(df: pd.DataFrame):
    """Min-Max scale the dataframe; return scaled frame + fitted scaler."""
    if df.empty:
        raise ValueError("Cannot normalize: dataframe is empty.")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Cannot normalize: dataframe contains NaN/inf values.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    print("\n  Normalization : DONE")
    print("  Features      :", list(df.columns))
    return df_scaled, scaler


def create_sequences(data: pd.DataFrame, target_col: str, window_size: int = 30):
    """Build sliding-window (X, y) arrays in one vectorised pass."""
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset.")
    if len(data) <= window_size:
        raise ValueError(
            f"Not enough rows ({len(data)}) for window_size={window_size}. "
            "Use a smaller window or more data."
        )

    values = data.values.astype(np.float32)
    target_idx = list(data.columns).index(target_col)
    n = len(values) - window_size

    # Vectorised stacking – avoids Python loop overhead
    X = np.lib.stride_tricks.sliding_window_view(values, (window_size, values.shape[1]))
    X = X[:n, 0, :, :]                          # shape (n, window, features)
    y = values[window_size: window_size + n, target_idx]

    print(f"  Time-series windowing : window={window_size}, samples={len(X)}")
    return X, y


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 – MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

def _rnn_block(layer_cls, units_1, units_2, input_shape):
    return Sequential([
        layer_cls(units_1, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        layer_cls(units_2),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])


def build_model(model_type: str, input_shape):
    """Factory that returns a compiled Keras model."""
    map_ = {"lstm": LSTM, "gru": GRU, "rnn": SimpleRNN}
    if model_type not in map_:
        raise ValueError(f"model_type must be one of {list(map_)}.")
    model = _rnn_block(map_[model_type], 64, 32, input_shape)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


class SimpleARModel:
    """Ridge-regression fallback when TensorFlow is unavailable."""

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        n, w, f = X.shape
        self.reg = Ridge()
        self.reg.fit(X.reshape(n, w * f), y)
        return self

    def predict(self, X, **_):
        n, w, f = X.shape
        return self.reg.predict(X.reshape(n, w * f))


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 – DRIFT DETECTORS  (vectorised where possible)
# ════════════════════════════════════════════════════════════════════════════

class ADWINDetector:
    """Simplified ADWIN: split-window mean-shift test."""

    def __init__(self, delta: float = 0.002, window_size: int = 100):
        self.delta = delta
        self.window_size = window_size
        self._buf: list[float] = []
        self.drift_points: list[int] = []

    def update(self, error: float, idx: int) -> bool:
        self._buf.append(abs(error))
        if len(self._buf) < self.window_size:
            return False
        w = np.asarray(self._buf[-self.window_size:])
        half = self.window_size // 2
        diff = abs(w[:half].mean() - w[half:].mean())
        threshold = np.sqrt(np.log(4 * len(self._buf) / self.delta) / (2 * half))
        if diff > threshold:
            self.drift_points.append(idx)
            self._buf = list(w[half:])
            return True
        return False


class PageHinkleyDetector:
    """Page-Hinkley test for upward mean shifts."""

    def __init__(self, delta: float = 0.005, lambda_: float = 20):
        self.delta = delta
        self.lambda_ = lambda_
        self._reset()
        self.drift_points: list[int] = []

    def _reset(self):
        self.m = self.T = self.M = self.n = 0.0

    def update(self, error: float, idx: int) -> bool:
        self.n += 1
        self.m += (abs(error) - self.m) / self.n
        self.T += abs(error) - self.m - self.delta
        self.M = min(self.M, self.T)
        if (self.T - self.M) > self.lambda_:
            self.drift_points.append(idx)
            self._reset()
            return True
        return False


class DDMDetector:
    """DDM-style detector based on fraction of large errors."""

    def __init__(self, error_threshold: float = 0.08,
                 drift_scale: float = 3.0, min_instances: int = 50):
        self.thr = error_threshold
        self.drift_scale = drift_scale
        self.min_n = min_instances
        self._reset()
        self.drift_points: list[int] = []

    def _reset(self):
        self.n = self.p = self.s = 0.0
        self.p_min = self.s_min = np.inf

    def update(self, error: float, idx: int) -> bool:
        hit = float(abs(error) > self.thr)
        self.n += 1
        self.p += (hit - self.p) / self.n
        self.s = np.sqrt(self.p * (1 - self.p) / max(self.n, 1))
        if self.n < self.min_n:
            return False
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min, self.s_min = self.p, self.s
        if self.p + self.s > self.p_min + self.drift_scale * self.s_min:
            self.drift_points.append(idx)
            self._reset()
            return True
        return False


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 – ADAPTIVE RETRAINING
# ════════════════════════════════════════════════════════════════════════════

def adaptive_retrain(model, X_new, y_new, model_type: str):
    """Fine-tune model on a recent data window after drift."""
    if TF_AVAILABLE and model_type in ("lstm", "gru", "rnn"):
        model.fit(
            X_new, y_new, epochs=5, batch_size=32, verbose=0,
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        )
    else:
        model.fit(X_new, y_new)
    print("    → Model retrained on recent window")
    return model


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 – MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(filepath: str, target_col: str = "T2M",
                 window_size: int = 30, model_type: str = "lstm",
                 test_ratio: float = 0.2):

    # 1 · Load & preprocess
    df_raw = load_and_clean_data(filepath)
    df_scaled, scaler = normalize_data(df_raw)

    # 2 · Sequences & split
    X, y = create_sequences(df_scaled, target_col, window_size)
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"\n  Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    # 3 · Build & train
    print("\n" + "=" * 60)
    print(f"STEP 2: BASE MODEL  ({model_type.upper()})")
    print("=" * 60)

    if TF_AVAILABLE:
        model = build_model(model_type, (X_train.shape[1], X_train.shape[2]))
        history = model.fit(
            X_train, y_train,
            validation_split=0.1, epochs=30, batch_size=32,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1,
        )
    else:
        print("Model: SimpleAR (TF not available)")
        model = SimpleARModel().fit(X_train, y_train)
        history = None

    # 4 · Online drift monitoring
    print("\n" + "=" * 60)
    print("STEP 3: DRIFT DETECTION & ADAPTIVE RETRAINING")
    print("=" * 60)

    adwin = ADWINDetector(delta=0.35, window_size=60)
    ph    = PageHinkleyDetector(delta=0.002, lambda_=0.6)
    ddm   = DDMDetector(error_threshold=0.07, drift_scale=3.0, min_instances=40)

    min_votes, min_gap = 1, 60
    last_drift = -10 ** 9
    retrain_window = 400

    # ── Batched inference for speed ───────────────────────────────────────
    # Predict the entire test set in one forward pass, then stream detectors.
    if TF_AVAILABLE:
        all_preds = model.predict(X_test, batch_size=256, verbose=0).ravel()
    else:
        all_preds = model.predict(X_test).ravel()

    predictions, true_values, errors = [], [], []
    drift_events: list[int] = []
    retrain_count = 0
    detector_hits = {"ADWIN": 0, "Page-Hinkley": 0, "DDM": 0}
    # Cache predictions as list so we can update after retraining
    preds_list = list(all_preds)

    print("\nRunning online drift monitoring …")

    for i in range(len(X_test)):
        if i % 200 == 0:
            print(f"  Progress: {i}/{len(X_test)}")

        pred  = float(preds_list[i])
        true  = float(y_test[i])
        error = true - pred

        predictions.append(pred)
        true_values.append(true)
        errors.append(error)

        fired = []
        if adwin.update(error, i): detector_hits["ADWIN"]        += 1; fired.append("ADWIN")
        if ph.update(error, i):    detector_hits["Page-Hinkley"] += 1; fired.append("Page-Hinkley")
        if ddm.update(error, i):   detector_hits["DDM"]          += 1; fired.append("DDM")

        if len(fired) >= min_votes and (i - last_drift) >= min_gap:
            print(f"  [Drift @ step {i:4d}] detected by {', '.join(fired)}")
            drift_events.append(i)
            last_drift = i

            recent_X = np.concatenate([X_train[-retrain_window:], X_test[max(0, i - retrain_window): i]])
            recent_y = np.concatenate([y_train[-retrain_window:], y_test[max(0, i - retrain_window): i]])
            model = adaptive_retrain(model, recent_X, recent_y, model_type)
            retrain_count += 1

            # Re-predict the remaining test steps after retraining
            remaining = X_test[i + 1:]
            if len(remaining):
                if TF_AVAILABLE:
                    new_preds = model.predict(remaining, batch_size=256, verbose=0).ravel()
                else:
                    new_preds = model.predict(remaining).ravel()
                preds_list[i + 1:] = list(new_preds)

    print(f"\n  Drift events   : {len(drift_events)}")
    print(f"  Retraining runs: {retrain_count}")
    print(f"  Detector hits  : ADWIN={detector_hits['ADWIN']}, "
          f"PH={detector_hits['Page-Hinkley']}, DDM={detector_hits['DDM']}")

    # 5 · Inverse-transform & metrics
    target_idx = list(df_scaled.columns).index(target_col)
    n_features  = df_scaled.shape[1]

    def inv(vals):
        dummy = np.zeros((len(vals), n_features), dtype=np.float32)
        dummy[:, target_idx] = vals
        return scaler.inverse_transform(dummy)[:, target_idx]

    preds_orig = inv(np.array(predictions))
    trues_orig = inv(np.array(true_values))

    mse  = mean_squared_error(trues_orig, preds_orig)
    mae  = mean_absolute_error(trues_orig, preds_orig)
    rmse = np.sqrt(mse)
    r2   = r2_score(trues_orig, preds_orig)
    mape = np.mean(np.abs((trues_orig - preds_orig) / np.maximum(np.abs(trues_orig), 1e-8))) * 100
    accuracy = max(0.0, 100.0 - mape)

    print("\n" + "=" * 60)
    print("FINAL METRICS  (original scale)")
    print("=" * 60)
    for name, val, unit in [("MAE", mae, "°C"), ("RMSE", rmse, "°C"),
                             ("MSE", mse, ""), ("R²", r2, ""),
                             ("MAPE", mape, "%"), ("Accuracy (100-MAPE)", accuracy, "%")]:
        print(f"  {name:<22}: {val:.4f} {unit}")

    # 6 · Plot
    plot_results(trues_orig, preds_orig, errors, drift_events,
                 adwin, history, target_col, model_type)

    return model, scaler, {
        "mae": mae, "rmse": rmse, "mse": mse, "r2": r2,
        "mape": mape, "accuracy": accuracy,
        "drift_events": drift_events,
        "retrain_count": retrain_count,
        "detector_hits": detector_hits,
    }


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 – VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

def plot_results(trues, preds, errors, drift_events,
                 adwin, history, target_col, model_type):
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle(
        f"Adaptive DL Pipeline  |  {model_type.upper()}  |  target: {target_col}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    drift_kw = dict(color="red", alpha=0.35, linewidth=0.8)

    def vlines(ax):
        for d in drift_events:
            ax.axvline(d, **drift_kw)

    # (a) Prediction vs Actual
    ax = axes[0, 0]
    ax.plot(trues, label="Actual",    color="steelblue", lw=1.2, alpha=0.85)
    ax.plot(preds, label="Predicted", color="orangered", lw=1.0, alpha=0.85, ls="--")
    vlines(ax)
    ax.set(title="Actual vs Predicted", xlabel="Test step", ylabel=f"{target_col} (°C)")
    ax.legend(); ax.grid(alpha=0.3)

    # (b) Prediction errors
    ax = axes[0, 1]
    ax.plot(errors, color="purple", lw=0.8, alpha=0.7)
    ax.axhline(0, color="black", lw=0.8)
    vlines(ax)
    ax.set(title="Prediction errors over time", xlabel="Test step", ylabel="Error")
    ax.grid(alpha=0.3)

    # (c) Rolling MAE
    ax = axes[1, 0]
    win = 50
    abs_e = np.abs(errors)
    rolling_mae = [abs_e[max(0, i - win): i + 1].mean() for i in range(len(errors))]
    ax.plot(rolling_mae, color="darkorange", lw=1.2)
    vlines(ax)
    ax.set(title=f"Rolling MAE (window={win})", xlabel="Test step", ylabel="MAE")
    ax.grid(alpha=0.3)

    # (d) ADWIN buffer
    ax = axes[1, 1]
    ax.plot(adwin._buf, color="teal", lw=0.9, alpha=0.85)
    ax.set(title="ADWIN – current error window", xlabel="Steps in window", ylabel="|Error|")
    ax.grid(alpha=0.3)

    # (e) Drift timeline
    ax = axes[2, 0]
    if drift_events:
        ax.stem(drift_events, [1] * len(drift_events),
                linefmt="r-", markerfmt="ro", basefmt="k-")
    ax.set(title="Drift detection timeline", xlabel="Test step", ylabel="")
    ax.set_yticks([])
    ax.grid(alpha=0.3)

    # (f) Training loss
    ax = axes[2, 1]
    if history and hasattr(history, "history"):
        ax.plot(history.history["loss"],     label="Train", color="royalblue")
        ax.plot(history.history["val_loss"], label="Val",   color="orange")
        ax.set(title="Training loss", xlabel="Epoch", ylabel="MSE")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Training history\nnot available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Training loss")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "adaptive_dl_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show(block=False); plt.pause(0.1); plt.close(fig)
    print(f"\n  Figure saved → {out}")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    DATA_PATH = resolve_data_path("Minor project dataset.csv")

    print("Adaptive Deep Learning Pipeline for Climate Data")
    print("=" * 60)
    print(f"TensorFlow available : {TF_AVAILABLE}")
    if TF_AVAILABLE:
        print(f"TF version           : {tf.__version__}")
    print(f"Dataset              : {DATA_PATH}\n")

    models_to_run = ["lstm", "gru", "rnn"] if TF_AVAILABLE else ["simple_ar"]
    all_results   = []

    for name in models_to_run:
        print("\n" + "=" * 60)
        print(f"RUNNING MODEL: {name.upper()}")
        print("=" * 60)

        _, _, metrics = run_pipeline(
            filepath=DATA_PATH, target_col="T2M",
            window_size=30, model_type=name, test_ratio=0.2,
        )

        all_results.append({
            "Model":      name.upper(),
            "MAE":        metrics["mae"],
            "RMSE":       metrics["rmse"],
            "MSE":        metrics["mse"],
            "R2":         metrics["r2"],
            "MAPE_%":     metrics["mape"],
            "Accuracy_%": metrics["accuracy"],
            "ADWIN":      metrics["detector_hits"]["ADWIN"],
            "PH":         metrics["detector_hits"]["Page-Hinkley"],
            "DDM":        metrics["detector_hits"]["DDM"],
            "Drifts":     len(metrics["drift_events"]),
            "Retrains":   metrics["retrain_count"],
        })

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    results_df = pd.DataFrame(all_results).sort_values("RMSE")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = results_df.iloc[0]
    print(f"\nBest model: {best['Model']}  "
          f"RMSE={best['RMSE']:.4f}  MAE={best['MAE']:.4f}  "
          f"Accuracy={best['Accuracy_%']:.2f}%")
    print("Pipeline complete.")