import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

st.set_page_config(
    page_title="ATP & WTA ML Match Predictor Lab",
    page_icon="🎾",
    layout="wide",
)


# ==============================================================================
# CACHED DATA & MODEL LOADERS
# ==============================================================================
@st.cache_resource
def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
    return create_client(url, key)


@st.cache_data(ttl=600)
def load_players() -> pd.DataFrame:
    sb = get_supabase()
    all_rows = []
    start = 0
    step = 1000
    while True:
        res = sb.table("players").select("*").range(start, start + step - 1).execute()
        if not res.data:
            break
        all_rows.extend(res.data)
        if len(res.data) < step:
            break
        start += step
    df = pd.DataFrame(all_rows)
    return df.sort_values("elo_overall", ascending=False).reset_index(drop=True)


@st.cache_data
def load_leaderboard() -> pd.DataFrame:
    return pd.read_csv("models/leaderboard.csv")


@st.cache_data
def load_feature_cols() -> list:
    with open("models/feature_cols.json") as f:
        return json.load(f)


def label_to_filename(full_label: str) -> str:
    safe = (
        full_label.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace("%", "")
    )
    return f"models/saved/{safe}.joblib"


@st.cache_resource
def load_all_pipelines(labels: list) -> dict:
    loaded = {}
    for lbl in labels:
        path = label_to_filename(lbl)
        if os.path.exists(path):
            loaded[lbl] = joblib.load(path)
    return loaded


def predict_prob_single(pipeline, X_row: pd.DataFrame) -> float:
    if hasattr(pipeline, "predict_proba"):
        return float(pipeline.predict_proba(X_row)[:, 1][0])
    return float(np.clip(pipeline.predict(X_row)[0], 0.0, 1.0))


def build_matchup_row(p1: pd.Series, p2: pd.Series, tour: str, surface: str, best_of: int, feature_cols: list) -> pd.DataFrame:
    surf_col = f"elo_{surface.lower()}"
    p1_elo = float(p1["elo_overall"])
    p2_elo = float(p2["elo_overall"])
    p1_surf = float(p1[surf_col])
    p2_surf = float(p2[surf_col])

    elo_prob = 1.0 / (1.0 + 10.0 ** ((p2_elo - p1_elo) / 400.0))
    surf_prob = 1.0 / (1.0 + 10.0 ** ((p2_surf - p1_surf) / 400.0))

    row = {
        "elo_diff": p1_elo - p2_elo,
        "surf_elo_diff": p1_surf - p2_surf,
        "elo_win_prob": elo_prob,
        "surf_elo_win_prob": surf_prob,
        "p1_elo": p1_elo,
        "p2_elo": p2_elo,
        "p1_surf_elo": p1_surf,
        "p2_surf_elo": p2_surf,
        "is_atp": 1 if tour == "ATP" else 0,
        "best_of": best_of,
        "surf_hard": 1 if surface == "Hard" else 0,
        "surf_clay": 1 if surface == "Clay" else 0,
        "surf_grass": 1 if surface == "Grass" else 0,
        "p1_lefty": 1 if p1.get("hand") == "L" else 0,
        "p2_lefty": 1 if p2.get("hand") == "L" else 0,
    }
    return pd.DataFrame([row])[feature_cols]


# ==============================================================================
# MAIN UI
# ==============================================================================
st.title("🎾 ATP & WTA Machine Learning Match Predictor Lab")
st.caption("Chronological Surface-Elo Engine • Multi-Model Benchmark (Raw vs. PCA) • Live Supabase Database")

players_df = load_players()
leaderboard_df = load_leaderboard()
feature_cols = load_feature_cols()
pipelines = load_all_pipelines(leaderboard_df["full_label"].tolist())

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Match Predictor",
    "🏆 Model Leaderboard & Overfitting Lab",
    "📈 Diagnostic Graphs & Learning Curves",
    "📊 ATP / WTA Elo Rankings",
])

# ------------------------------------------------------------------------------
# TAB 1: LIVE MATCH PREDICTOR (ALL MODELS SIDE-BY-SIDE)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Head-to-Head Matchup Simulator")

    c1, c2, c3, c4 = st.columns(4)
    tour = c1.selectbox("Tour", ["ATP", "WTA"])
    surface = c2.selectbox("Court Surface", ["Hard", "Clay", "Grass"])
    best_of = c3.selectbox("Match Format", [3, 5], index=0, format_func=lambda x: f"Best of {x} Sets")
    primary_model = c4.selectbox("Primary Model", leaderboard_df["full_label"].tolist(), index=0)

    tour_players = players_df[players_df["tour"] == tour].reset_index(drop=True)
    player_names = tour_players["player_name"].tolist()

    col_p1, col_p2 = st.columns(2)
    p1_name = col_p1.selectbox("Player 1", player_names, index=0)
    p2_name = col_p2.selectbox("Player 2", player_names, index=min(1, len(player_names) - 1))

    if p1_name == p2_name:
        st.warning("Select two different players to simulate a match.")
    else:
        p1_row = tour_players[tour_players["player_name"] == p1_name].iloc[0]
        p2_row = tour_players[tour_players["player_name"] == p2_name].iloc[0]
        X_match = build_matchup_row(p1_row, p2_row, tour, surface, best_of, feature_cols)

        # Primary model prediction
        p1_prob = predict_prob_single(pipelines[primary_model], X_match)
        p2_prob = 1.0 - p1_prob

        st.markdown("---")
        m1, m2, m3 = st.columns([2, 1, 2])
        with m1:
            st.metric(
                label=f"{p1_name} (Overall Elo: {p1_row['elo_overall']:.0f} | {surface}: {p1_row[f'elo_{surface.lower()}']:.0f})",
                value=f"{p1_prob:.1%} Win Chance",
            )
        with m2:
            st.markdown("<h3 style='text-align: center; padding-top: 15px;'>VS</h3>", unsafe_allow_html=True)
        with m3:
            st.metric(
                label=f"{p2_name} (Overall Elo: {p2_row['elo_overall']:.0f} | {surface}: {p2_row[f'elo_{surface.lower()}']:.0f})",
                value=f"{p2_prob:.1%} Win Chance",
            )

        # All 12 Models Consensus Chart
        st.subheader("How All 12 Trained Models Vote on This Match")
        model_votes = []
        for _, lb_row in leaderboard_df.iterrows():
            lbl = lb_row["full_label"]
            prob1 = predict_prob_single(pipelines[lbl], X_match)
            model_votes.append({
                "Model": lbl,
                "Feature Mode": lb_row["feature_mode"],
                f"{p1_name} Win %": round(prob1 * 100, 2),
                f"{p2_name} Win %": round((1 - prob1) * 100, 2),
                "Model Test AUC": lb_row["test_auc"],
            })
        votes_df = pd.DataFrame(model_votes)

        fig_votes = px.bar(
            votes_df,
            x=f"{p1_name} Win %",
            y="Model",
            color="Feature Mode",
            orientation="h",
            range_x=[0, 100],
            text=f"{p1_name} Win %",
            title=f"Predicted Win Probability for {p1_name} Across Every Model",
        )
        fig_votes.add_vline(x=50, line_dash="dash", line_color="gray")
        fig_votes.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
        st.plotly_chart(fig_votes, use_container_width=True)

        with st.expander("Inspect Exact Matchup Feature Vector Fed into Models"):
            st.dataframe(X_match, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: MODEL LEADERBOARD & OVERFITTING LAB
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Train (2016–2024) vs. Out-of-Sample Test (2025–2026) Benchmark")

    mode_filter = st.multiselect(
        "Filter by Feature Pipeline",
        options=leaderboard_df["feature_mode"].unique().tolist(),
        default=leaderboard_df["feature_mode"].unique().tolist(),
    )
    filtered_lb = leaderboard_df[leaderboard_df["feature_mode"].isin(mode_filter)].copy()

    # Interactive Train vs Test Accuracy Chart
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(
        name="Train Accuracy (2016-2024)",
        x=filtered_lb["full_label"],
        y=filtered_lb["train_acc"] * 100,
        marker_color="#4C72B0",
    ))
    fig_gap.add_trace(go.Bar(
        name="Out-of-Sample Test Accuracy (2025-2026)",
        x=filtered_lb["full_label"],
        y=filtered_lb["test_acc"] * 100,
        marker_color="#55A868",
    ))
    fig_gap.update_layout(
        barmode="group",
        title="Train vs. Test Accuracy (Spot Overfitting When Blue Bar Towers Over Green Bar)",
        yaxis_title="Accuracy (%)",
        yaxis_range=[58, 72],
        height=430,
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    st.dataframe(
        filtered_lb.style.format({
            "train_acc": "{:.2%}",
            "test_acc": "{:.2%}",
            "overfit_gap": "{:+.2%}",
            "train_auc": "{:.4f}",
            "test_auc": "{:.4f}",
            "train_brier": "{:.4f}",
            "test_brier": "{:.4f}",
            "train_seconds": "{:.2f}s",
        }),
        use_container_width=True,
    )

# ------------------------------------------------------------------------------
# TAB 3: DIAGNOSTIC GRAPHS & LEARNING CURVES
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Model Diagnostics: Learning Curves, ROC Curves & PCA Variance")

    plot_specs = [
        ("models/plots/5_learning_curve.png", "1. Training vs. Validation Learning Curve",
         "Shows how model accuracy scales as we feed more historical matches (from 15% up to 100% of the training set). Converging curves indicate healthy generalization."),
        ("models/plots/3_roc_curves.png", "2. Out-of-Sample ROC Curves (2025–2026)",
         "Compares True Positive Rate vs. False Positive Rate across all probability thresholds. Higher Area Under the Curve (AUC) means better ranking separation."),
        ("models/plots/2_pca_vs_raw_auc.png", "3. Raw Features vs. PCA (95% Variance) Impact",
         "Compares each algorithm before and after Principal Component Analysis. Linear models benefit from decorrelated components, while tree models prefer raw splits."),
        ("models/plots/4_pca_variance.png", "4. PCA Scree & Cumulative Explained Variance Plot",
         "Shows how much information each principal component captures and how many components are needed to hit the 95% variance threshold."),
        ("models/plots/1_train_vs_test_overfit.png", "5. Train vs. Test Overfitting Gap",
         "Highlights models that memorized historical noise (like Random Forest) versus models that stayed balanced on unseen 2025–2026 matches."),
    ]

    for img_path, title, desc in plot_specs:
        if os.path.exists(img_path):
            st.markdown(f"### {title}")
            st.write(desc)
            st.image(img_path, use_container_width=True)
            st.markdown("---")

# ------------------------------------------------------------------------------
# TAB 4: ATP / WTA ELO RANKINGS EXPLORER
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Live Supabase Player Elo Ratings (2015–2026)")
    f1, f2, f3 = st.columns([1, 1, 2])
    tour_pick = f1.selectbox("Filter Tour", ["ALL", "ATP", "WTA"])
    min_matches = f2.slider("Minimum Matches Played", 0, 200, 30)
    search_q = f3.text_input("Search Player Name", "")

    view_df = players_df[players_df["matches_played"] >= min_matches].copy()
    if tour_pick != "ALL":
        view_df = view_df[view_df["tour"] == tour_pick]
    if search_q.strip():
        view_df = view_df[view_df["player_name"].str.contains(search_q.strip(), case=False, na=False)]

    view_df = view_df.reset_index(drop=True)
    view_df.index += 1
    st.dataframe(
        view_df[["player_name", "tour", "hand", "matches_played", "elo_overall", "elo_hard", "elo_clay", "elo_grass"]],
        use_container_width=True,
    )