import json
import os
import time
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ==============================================================================
# 1. EASY CONFIGURATION (Edit hyperparameters or add/remove models here!)
# ==============================================================================
TEST_PCA_COMPARISON = True   # Runs every model both WITHOUT PCA and WITH PCA
PCA_VARIANCE_KEEP = 0.95     # Keep 95% of variance when PCA is enabled

MODEL_REGISTRY = {
    "Linear Regression (LPM)": LinearRegression(),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1
    ),
    "XGBoost Classifier": XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        eval_metric="logloss", random_state=42, n_jobs=-1
    ),
    "XGBoost Regressor": XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        random_state=42, n_jobs=-1
    ),
    "LightGBM Classifier": LGBMClassifier(
        n_estimators=400, max_depth=4, num_leaves=15, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        random_state=42, verbose=-1, n_jobs=-1
    ),
}


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
def elo_win_prob(elo_a: np.ndarray, elo_b: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    df = df[df["tourney_date"].dt.year >= 2016].reset_index(drop=True)

    swap = (df.index % 2 == 1)

    p1_elo = np.where(swap, df["l_pre_elo"], df["w_pre_elo"])
    p2_elo = np.where(swap, df["w_pre_elo"], df["l_pre_elo"])
    p1_surf_elo = np.where(swap, df["l_pre_surf_elo"], df["w_pre_surf_elo"])
    p2_surf_elo = np.where(swap, df["w_pre_surf_elo"], df["l_pre_surf_elo"])

    w_hand = df.get("winner_hand", pd.Series("R", index=df.index)).fillna("R")
    l_hand = df.get("loser_hand", pd.Series("R", index=df.index)).fillna("R")
    p1_hand = np.where(swap, l_hand, w_hand)
    p2_hand = np.where(swap, w_hand, l_hand)
    surf = df["surface"].fillna("Hard")

    return pd.DataFrame({
        "tourney_date": df["tourney_date"],
        "elo_diff": p1_elo - p2_elo,
        "surf_elo_diff": p1_surf_elo - p2_surf_elo,
        "elo_win_prob": elo_win_prob(p1_elo, p2_elo),
        "surf_elo_win_prob": elo_win_prob(p1_surf_elo, p2_surf_elo),
        "p1_elo": p1_elo,
        "p2_elo": p2_elo,
        "p1_surf_elo": p1_surf_elo,
        "p2_surf_elo": p2_surf_elo,
        "is_atp": (df["tour"] == "ATP").astype(int),
        "best_of": pd.to_numeric(df["best_of"], errors="coerce").fillna(3),
        "surf_hard": (surf == "Hard").astype(int),
        "surf_clay": (surf == "Clay").astype(int),
        "surf_grass": (surf == "Grass").astype(int),
        "p1_lefty": (p1_hand == "L").astype(int),
        "p2_lefty": (p2_hand == "L").astype(int),
        "target": (~swap).astype(int),
    })


# ==============================================================================
# 3. TRAINING, EVALUATION & OVERFITTING DIAGNOSTICS
# ==============================================================================
def predict_probabilities(estimator, X: pd.DataFrame) -> np.ndarray:
    """Handles both Classifiers (predict_proba) and Regressors (clipped predict)."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    raw = estimator.predict(X)
    return np.clip(raw, 0.0, 1.0)


def diagnose_fit(train_acc: float, test_acc: float) -> str:
    gap = train_acc - test_acc
    if gap > 0.035:
        return "Overfitting (High Train-Test Gap)"
    if train_acc < 0.625 and test_acc < 0.625:
        return "Underfitting (Low Capacity)"
    return "Well-Balanced"


def run_experiments():
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("models/plots", exist_ok=True)

    print("Loading historical Elo dataset...")
    raw_df = pd.read_csv("data/historical_matches_elo.csv", low_memory=False)
    data = build_feature_matrix(raw_df)
    feature_cols = [c for c in data.columns if c not in ["tourney_date", "target"]]

    train_df = data[data["tourney_date"].dt.year <= 2024]
    test_df = data[data["tourney_date"].dt.year >= 2025]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    print(f"Train Set (2016-2024): {len(X_train):,} matches | Test Set (2025-2026): {len(X_test):,} matches\n")

    pca_modes = [False, True] if TEST_PCA_COMPARISON else [False]
    results = []
    roc_curves_data = {}

    for use_pca in pca_modes:
        mode_label = "With PCA (95%)" if use_pca else "Raw Features"
        for name, base_model in MODEL_REGISTRY.items():
            # Fresh clone of estimator
            from sklearn.base import clone
            est = clone(base_model)

            if use_pca:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=PCA_VARIANCE_KEEP, random_state=42)),
                    ("model", est),
                ])
            else:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", est),
                ])

            t0 = time.time()
            pipeline.fit(X_train, y_train)
            train_time = round(time.time() - t0, 2)

            train_probs = predict_probabilities(pipeline, X_train)
            test_probs = predict_probabilities(pipeline, X_test)

            train_acc = accuracy_score(y_train, (train_probs >= 0.5).astype(int))
            test_acc = accuracy_score(y_test, (test_probs >= 0.5).astype(int))

            train_auc = roc_auc_score(y_train, train_probs)
            test_auc = roc_auc_score(y_test, test_probs)

            train_brier = brier_score_loss(y_train, train_probs)
            test_brier = brier_score_loss(y_test, test_probs)

            fit_status = diagnose_fit(train_acc, test_acc)
            full_label = f"{name} [{mode_label}]"

            results.append({
                "model_name": name,
                "feature_mode": mode_label,
                "full_label": full_label,
                "train_acc": round(train_acc, 4),
                "test_acc": round(test_acc, 4),
                "overfit_gap": round(train_acc - test_acc, 4),
                "train_auc": round(train_auc, 4),
                "test_auc": round(test_auc, 4),
                "train_brier": round(train_brier, 4),
                "test_brier": round(test_brier, 4),
                "fit_status": fit_status,
                "train_seconds": train_time,
            })

            if not use_pca:
                fpr, tpr, _ = roc_curve(y_test, test_probs)
                roc_curves_data[name] = (fpr, tpr, test_auc)

            # Save every pipeline so the Super Dashboard can load any model live
            safe_fname = full_label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("%", "")
            joblib.dump(pipeline, f"models/saved/{safe_fname}.joblib")

            print(
                f"{full_label:<45} | Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%} "
                f"| Gap: {train_acc - test_acc:+.2%} | AUC: {test_auc:.4f} | {fit_status}"
            )

    leaderboard = pd.DataFrame(results).sort_values(["test_auc", "test_acc"], ascending=False)
    leaderboard.to_csv("models/leaderboard.csv", index=False)
    with open("models/feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("\nGenerating diagnostic plots for Super Dashboard...")
    generate_diagnostic_plots(leaderboard, roc_curves_data, X_train, y_train, feature_cols)
    print("Done! Saved leaderboard to models/leaderboard.csv and charts to models/plots/")


# ==============================================================================
# 4. DIAGNOSTIC CHARTS (Saved for Jupyter & Streamlit Dashboard)
# ==============================================================================
def generate_diagnostic_plots(leaderboard, roc_curves_data, X_train, y_train, feature_cols):
    # Plot 1: Train vs Test Accuracy (Overfitting / Underfitting Check)
    raw_lb = leaderboard[leaderboard["feature_mode"] == "Raw Features"].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(raw_lb))
    width = 0.36
    ax.bar(x - width / 2, raw_lb["train_acc"] * 100, width, label="Train Accuracy (2016-2024)", color="#4C72B0")
    ax.bar(x + width / 2, raw_lb["test_acc"] * 100, width, label="Test Accuracy (2025-2026)", color="#55A868")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overfitting vs. Generalization: Train vs. Out-of-Sample Test Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(raw_lb["model_name"], rotation=20, ha="right")
    ax.set_ylim(55, 80)
    ax.legend()
    plt.tight_layout()
    plt.savefig("models/plots/1_train_vs_test_overfit.png", dpi=150)
    plt.close()

    # Plot 2: Raw Features vs PCA Test ROC-AUC Comparison
    if TEST_PCA_COMPARISON:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=leaderboard, x="model_name", y="test_auc", hue="feature_mode", ax=ax)
        ax.set_title("Impact of PCA (95% Variance) on Out-of-Sample Test ROC-AUC")
        ax.set_ylabel("Test ROC-AUC")
        ax.set_xlabel("")
        ax.set_ylim(0.65, 0.74)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig("models/plots/2_pca_vs_raw_auc.png", dpi=150)
        plt.close()

    # Plot 3: Multi-Model ROC Curves (Test Set)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr, auc_val) in roc_curves_data.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Coin Flip (0.5000)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Out-of-Sample ROC Curves (2025-2026 Matches)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("models/plots/3_roc_curves.png", dpi=150)
    plt.close()

    # Plot 4: PCA Explained Variance Scree Plot
    scaler = StandardScaler()
    pca = PCA().fit(scaler.fit_transform(X_train))
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(1, len(cum_var) + 1), pca.explained_variance_ratio_ * 100, alpha=0.6, label="Individual Component %")
    ax.plot(range(1, len(cum_var) + 1), cum_var, "ro-", label="Cumulative Variance %")
    ax.axhline(95, color="green", linestyle="--", label="95% Threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Explained Variance Across Tennis Features")
    ax.legend()
    plt.tight_layout()
    plt.savefig("models/plots/4_pca_variance.png", dpi=150)
    plt.close()

    # Plot 5: Learning Curve (Train vs Validation Score as Dataset Grows)
    tscv = TimeSeriesSplit(n_splits=5)
    train_sizes, train_scores, val_scores = learning_curve(
        XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.03, random_state=42, n_jobs=-1),
        X_train, y_train, cv=tscv, scoring="accuracy",
        train_sizes=np.linspace(0.15, 1.0, 6), n_jobs=-1
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_sizes, train_scores.mean(axis=1) * 100, "o-", color="#4C72B0", label="Training Score")
    ax.plot(train_sizes, val_scores.mean(axis=1) * 100, "o-", color="#DD8452", label="Chronological Validation Score")
    ax.set_xlabel("Number of Training Matches")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Learning Curve: Checking Underfitting vs. Overfitting Across Sample Sizes")
    ax.legend()
    plt.tight_layout()
    plt.savefig("models/plots/5_learning_curve.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_experiments()