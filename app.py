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


@st.cache_data(ttl=600)
def load_player_matches(player_uid: str) -> pd.DataFrame:
    """Fetches all career matches (2015-present) for one player from Supabase."""
    sb = get_supabase()
    all_rows = []
    start = 0
    step = 1000
    while True:
        res = (
            sb.table("matches")
            .select("*")
            .or_(f"winner_uid.eq.{player_uid},loser_uid.eq.{player_uid}")
            .order("tourney_date", desc=True)
            .range(start, start + step - 1)
            .execute()
        )
        if not res.data:
            break
        all_rows.extend(res.data)
        if len(res.data) < step:
            break
        start += step

    if not all_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(all_rows)
    won = raw["winner_uid"] == player_uid

    formatted = pd.DataFrame({
        "Date": raw["tourney_date"].astype(str).str[:10],
        "Result": np.where(won, "✅ W", "❌ L"),
        "Won": won,
        "Opponent": np.where(won, raw["loser_name"], raw["winner_name"]),
        "Opponent_UID": np.where(won, raw["loser_uid"], raw["winner_uid"]),
        "Score": raw["score"].fillna(""),
        "Tournament": raw["tourney_name"].fillna(""),
        "Surface": raw["surface"].fillna("Hard"),
        "Round": raw["round"].fillna(""),
        "Player Elo": pd.to_numeric(np.where(won, raw["w_pre_elo"], raw["l_pre_elo"]), errors="coerce"),
        "Opp Elo": pd.to_numeric(np.where(won, raw["l_pre_elo"], raw["w_pre_elo"]), errors="coerce"),
    })
    return formatted.sort_values("Date", ascending=False).reset_index(drop=True)


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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Live Match Predictor",
    "🏆 Model Leaderboard & Overfitting Lab",
    "📈 Diagnostic Graphs & Learning Curves",
    "📊 ATP / WTA Elo Rankings",
    "📋 Player Match History",
])

# ------------------------------------------------------------------------------
# TAB 1: LIVE MATCH PREDICTOR (ALL MODELS SIDE-BY-SIDE + RECENT FORM)
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

        # Recent Form & Head-to-Head Preview
        p1_hist = load_player_matches(p1_row["player_id"])
        p2_hist = load_player_matches(p2_row["player_id"])

        if not p1_hist.empty and not p2_hist.empty:
            h2h = p1_hist[p1_hist["Opponent_UID"] == p2_row["player_id"]]
            p1_h2h_wins = int(h2h["Won"].sum()) if not h2h.empty else 0
            p2_h2h_wins = len(h2h) - p1_h2h_wins

            st.subheader(f"Recent Form & Head-to-Head ({p1_name} {p1_h2h_wins} – {p2_h2h_wins} {p2_name})")
            if not h2h.empty:
                with st.expander(f"View All {len(h2h)} Head-to-Head Meetings (2015–Present)"):
                    st.dataframe(
                        h2h[["Date", "Result", "Score", "Tournament", "Surface", "Round"]],
                        use_container_width=True,
                        hide_index=True,
                    )

            rc1, rc2 = st.columns(2)
            with rc1:
                last10_p1 = p1_hist.head(10)
                w10_1 = int(last10_p1["Won"].sum())
                st.markdown(f"**{p1_name} — Latest 10 Matches ({w10_1}W – {len(last10_p1) - w10_1}L)**")
                st.dataframe(
                    last10_p1[["Date", "Result", "Opponent", "Score", "Surface", "Tournament"]],
                    use_container_width=True,
                    hide_index=True,
                )
            with rc2:
                last10_p2 = p2_hist.head(10)
                w10_2 = int(last10_p2["Won"].sum())
                st.markdown(f"**{p2_name} — Latest 10 Matches ({w10_2}W – {len(last10_p2) - w10_2}L)**")
                st.dataframe(
                    last10_p2[["Date", "Result", "Opponent", "Score", "Surface", "Tournament"]],
                    use_container_width=True,
                    hide_index=True,
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

# ------------------------------------------------------------------------------
# TAB 5: PLAYER MATCH HISTORY & ELO EVOLUTION
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("Player Match History & Elo Progression (2015–Present)")

    hc1, hc2, hc3, hc4 = st.columns([1, 2, 1, 1])
    hist_tour = hc1.selectbox("Tour", ["ATP", "WTA"], key="hist_tour")
    hist_tour_players = players_df[players_df["tour"] == hist_tour].reset_index(drop=True)
    hist_player_name = hc2.selectbox("Select Player", hist_tour_players["player_name"].tolist(), key="hist_player")
    hist_limit = hc3.selectbox("Show Games", ["Latest 15", "Latest 50", "All Career Matches"], index=0)
    hist_surface = hc4.selectbox("Surface Filter", ["ALL", "Hard", "Clay", "Grass"], key="hist_surf")

    sel_player = hist_tour_players[hist_tour_players["player_name"] == hist_player_name].iloc[0]
    match_log = load_player_matches(sel_player["player_id"])

    if match_log.empty:
        st.info("No matches found for this player yet.")
    else:
        # Summary KPIs
        total_w = int(match_log["Won"].sum())
        total_l = len(match_log) - total_w
        win_pct = (total_w / len(match_log)) * 100 if len(match_log) else 0.0
        last10 = match_log.head(10)
        l10_w = int(last10["Won"].sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Career Record (2015–Now)", f"{total_w}W – {total_l}L", f"{win_pct:.1f}% Win Rate")
        k2.metric("Latest 10 Matches", f"{l10_w}W – {len(last10) - l10_w}L")
        k3.metric("Current Overall Elo", f"{sel_player['elo_overall']:.0f}")
        k4.metric(
            "Surface Elo (H / C / G)",
            f"{sel_player['elo_hard']:.0f} / {sel_player['elo_clay']:.0f} / {sel_player['elo_grass']:.0f}",
        )

        # Elo Progression Chart
        elo_series = match_log.dropna(subset=["Player Elo"]).iloc[::-1].copy()
        if len(elo_series) > 1:
            fig_elo = px.line(
                elo_series,
                x="Date",
                y="Player Elo",
                hover_data=["Result", "Opponent", "Score", "Tournament", "Surface"],
                title=f"{hist_player_name} — Pre-Match Elo Rating Over Time",
            )
            fig_elo.update_traces(line_color="#1f77b4", line_width=2)
            fig_elo.update_layout(height=320, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig_elo, use_container_width=True)

        # Filter & Display Match Table
        sf1, sf2 = st.columns([1, 2])
        result_filter = sf1.radio("Result Filter", ["All", "Wins Only", "Losses Only"], horizontal=True)
        opp_search = sf2.text_input("Filter by Opponent or Tournament", "", key="hist_search")

        filtered_log = match_log.copy()
        if hist_surface != "ALL":
            filtered_log = filtered_log[filtered_log["Surface"] == hist_surface]
        if result_filter == "Wins Only":
            filtered_log = filtered_log[filtered_log["Won"]]
        elif result_filter == "Losses Only":
            filtered_log = filtered_log[~filtered_log["Won"]]
        if opp_search.strip():
            q = opp_search.strip()
            filtered_log = filtered_log[
                filtered_log["Opponent"].str.contains(q, case=False, na=False)
                | filtered_log["Tournament"].str.contains(q, case=False, na=False)
            ]

        if hist_limit == "Latest 15":
            filtered_log = filtered_log.head(15)
        elif hist_limit == "Latest 50":
            filtered_log = filtered_log.head(50)

        st.caption(f"Showing {len(filtered_log)} matches")
        st.dataframe(
            filtered_log[["Date", "Result", "Opponent", "Score", "Tournament", "Surface", "Round", "Player Elo", "Opp Elo"]],
            use_container_width=True,
            hide_index=True,
        )
