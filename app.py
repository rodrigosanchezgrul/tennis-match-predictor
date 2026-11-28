import json
import math
import os
import re
import urllib.request
from datetime import datetime, timedelta, timezone
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


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _calc_age(birth_date_val) -> float:
    if not birth_date_val or pd.isna(birth_date_val):
        return 26.0
    try:
        bd = datetime.strptime(str(birth_date_val)[:10], "%Y-%m-%d").date()
        return round((datetime.now(timezone.utc).date() - bd).days / 365.25, 1)
    except Exception:
        return 26.0


def _infer_surface(tourney_name: str) -> str:
    name = (tourney_name or "").lower()
    clay = [
        "roland garros", "french open", "madrid", "rome", "internazionali",
        "monte-carlo", "monte carlo", "barcelona", "hamburg", "bastad",
        "gstaad", "kitzbuhel", "umag", "palermo", "rabat", "strasbourg",
        "charleston", "bogota", "rio", "buenos aires", "santiago", "estoril",
        "lyon", "geneva", "munich", "marrakech", "bucharest", "houston", "iasi",
    ]
    grass = [
        "wimbledon", "halle", "queen", "eastbourne", "mallorca",
        "hertogenbosch", "stuttgart", "bad homburg", "nottingham", "birmingham", "berlin",
    ]
    if any(k in name for k in clay):
        return "Clay"
    if any(k in name for k in grass):
        return "Grass"
    return "Hard"


def _infer_level(tourney_name: str) -> str:
    name = (tourney_name or "").lower()
    if any(g in name for g in ["australian open", "roland garros", "french open", "wimbledon", "us open"]):
        return "G"
    if any(m in name for m in ["masters", "1000", "indian wells", "miami", "madrid", "rome", "canada", "cincinnati", "shanghai", "paris", "doha", "dubai", "beijing", "wuhan"]):
        return "M"
    return "A"


def _is_valid_singles_draw(draw_name: str, tour_label: str) -> bool:
    d = (draw_name or "").lower()
    if "singles" not in d:
        return False
    if any(x in d for x in ["wheelchair", "junior", "boy", "girl", "quad", "legend"]):
        return False
    if tour_label == "ATP" and "women" in d:
        return False
    if tour_label == "WTA" and re.search(r"\bmen('s)?\b", d):
        return False
    return True


@st.cache_data(ttl=300)
def fetch_upcoming_espn_matches() -> pd.DataFrame:
    now_utc = datetime.now(timezone.utc)
    end_utc = now_utc + timedelta(days=2)
    date_range = f"{now_utc.strftime('%Y%m%d')}-{end_utc.strftime('%Y%m%d')}"

    rows = []
    seen = set()

    for slug, tour_label in [("atp", "ATP"), ("wta", "WTA")]:
        url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/{slug}/scoreboard?dates={date_range}&limit=100"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue

        for event in payload.get("events", []):
            t_name = event.get("name", f"{tour_label} Event")
            surface = _infer_surface(t_name)
            level = _infer_level(t_name)
            best_of = 5 if (tour_label == "ATP" and level == "G") else 3

            comps = []
            groups = event.get("groupings", [])
            if groups:
                for grp in groups:
                    draw_name = grp.get("grouping", {}).get("displayName", "")
                    if not _is_valid_singles_draw(draw_name, tour_label):
                        continue
                    comps.extend(grp.get("competitions", []))
            else:
                comps.extend(event.get("competitions", []))

            for comp in comps:
                status_type = comp.get("status", {}).get("type", {})
                if status_type.get("completed", False):
                    continue
                competitors = comp.get("competitions") or comp.get("competitors", [])
                if len(competitors) != 2:
                    continue

                p1_name = (competitors[0].get("athlete") or competitors[0].get("team") or {}).get("displayName", "")
                p2_name = (competitors[1].get("athlete") or competitors[1].get("team") or {}).get("displayName", "")
                if not p1_name or not p2_name or "TBD" in (p1_name, p2_name):
                    continue

                dedup = (tour_label, _norm_name(p1_name), _norm_name(p2_name))
                if dedup in seen:
                    continue
                seen.add(dedup)

                raw_dt = comp.get("date") or ""
                try:
                    dt_obj = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                    time_label = dt_obj.strftime("%b %d • %H:%M UTC")
                    sort_ts = dt_obj.timestamp()
                except Exception:
                    time_label = status_type.get("shortDetail") or "Scheduled"
                    sort_ts = 0.0

                state = status_type.get("state", "pre")
                status_badge = "🔴 LIVE" if state == "in" else f"🕒 {time_label}"

                rows.append({
                    "sort_ts": sort_ts,
                    "Status": status_badge,
                    "IsLive": state == "in",
                    "Tour": tour_label,
                    "Tournament": t_name,
                    "Surface": surface,
                    "Round": (comp.get("round") or {}).get("displayName") or "",
                    "BestOf": best_of,
                    "Player1": p1_name,
                    "Player2": p2_name,
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["IsLive", "sort_ts"], ascending=[False, True]).reset_index(drop=True)


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
    for col, default in [
        ("ranking_points", 0), ("grand_slams", 0), ("masters_1000", 0),
        ("total_titles", 0), ("form_10", 0.5), ("matches_14d", 0),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    df["age"] = df["birth_date"].apply(_calc_age) if "birth_date" in df.columns else 26.0
    return df.sort_values("elo_overall", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=600)
def load_player_matches(player_uid: str) -> pd.DataFrame:
    if not player_uid:
        return pd.DataFrame()
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
        .replace(" ", "_").replace("(", "").replace(")", "")
        .replace("[", "").replace("]", "").replace("%", "")
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


def explain_matchup_drivers(pipeline, X_row: pd.DataFrame, p1, p2, p1_name: str, p2_name: str, surface: str, p1_hist: pd.DataFrame, p2_hist: pd.DataFrame):
    """Computes per-match feature impact (% win prob shift) + Head-to-Head & Overall Win Rate comparison."""
    base_prob = predict_prob_single(pipeline, X_row)
    surf_col = f"elo_{surface.lower()}"

    # Neutralize each conceptual group to parity (both players equal to midpoint) to measure exact local impact
    mid_elo = (float(p1["elo_overall"]) + float(p2["elo_overall"])) / 2.0
    mid_surf = (float(p1[surf_col]) + float(p2[surf_col])) / 2.0
    mid_age = (float(p1.get("age", 26.0)) + float(p2.get("age", 26.0))) / 2.0

    groups = [
        (
            "Overall Elo Rating",
            {"elo_diff": 0.0, "elo_win_prob": 0.5, "p1_elo": mid_elo, "p2_elo": mid_elo},
            f"{p1['elo_overall']:.0f}",
            f"{p2['elo_overall']:.0f}",
        ),
        (
            f"{surface} Court Elo",
            {"surf_elo_diff": 0.0, "surf_elo_win_prob": 0.5, "p1_surf_elo": mid_surf, "p2_surf_elo": mid_surf},
            f"{p1[surf_col]:.0f}",
            f"{p2[surf_col]:.0f}",
        ),
        (
            "52-Week Ranking Points",
            {"pts_diff": 0.0, "log_pts_ratio": 0.0},
            f"{int(p1.get('ranking_points', 0)):,} pts",
            f"{int(p2.get('ranking_points', 0)):,} pts",
        ),
        (
            "Career Titles (GS / M1000)",
            {"slam_diff": 0.0, "masters_diff": 0.0, "titles_diff": 0.0},
            f"{int(p1.get('grand_slams', 0))} GS • {int(p1.get('masters_1000', 0))} M1000 ({int(p1.get('total_titles', 0))} tot)",
            f"{int(p2.get('grand_slams', 0))} GS • {int(p2.get('masters_1000', 0))} M1000 ({int(p2.get('total_titles', 0))} tot)",
        ),
        (
            "Player Age & Peak Curve",
            {"age_diff": 0.0, "p1_age": mid_age, "p2_age": mid_age, "p1_peak_dist": abs(mid_age - 26.0), "p2_peak_dist": abs(mid_age - 26.0)},
            f"{float(p1.get('age', 26.0)):.1f} yrs",
            f"{float(p2.get('age', 26.0)):.1f} yrs",
        ),
        (
            "Recent Form (Last 10)",
            {"form10_diff": 0.0},
            f"{float(p1.get('form_10', 0.5)):.0%} win rate",
            f"{float(p2.get('form_10', 0.5)):.0%} win rate",
        ),
        (
            "14-Day Match Load",
            {"fatigue14_diff": 0.0},
            f"{int(p1.get('matches_14d', 0))} matches",
            f"{int(p2.get('matches_14d', 0))} matches",
        ),
    ]

    # Direction sign (+1 if P1 strictly beats P2 on that metric, -1 if P2 beats P1, 0 if tied)
    true_sign = {
        "Overall Elo Rating": np.sign(float(p1["elo_overall"]) - float(p2["elo_overall"])),
        f"{surface} Court Elo": np.sign(float(p1[surf_col]) - float(p2[surf_col])),
        "52-Week Ranking Points": np.sign(float(p1.get("ranking_points", 0)) - float(p2.get("ranking_points", 0))),
        "Career Titles (GS / M1000)": np.sign((float(p1.get("grand_slams", 0)) * 3 + float(p1.get("masters_1000", 0))) - (float(p2.get("grand_slams", 0)) * 3 + float(p2.get("masters_1000", 0)))),
        "Player Age & Peak Curve": np.sign(abs(float(p2.get("age", 26.0)) - 25.0) - abs(float(p1.get("age", 26.0)) - 25.0)),
        "Recent Form (Last 10)": np.sign(float(p1.get("form_10", 0.5)) - float(p2.get("form_10", 0.5))),
        "14-Day Match Load": np.sign(float(p1.get("matches_14d", 0)) - float(p2.get("matches_14d", 0))),
    }

    driver_rows = []
    for label, overrides, p1_disp, p2_disp in groups:
        X_cf = X_row.copy()
        for col, val in overrides.items():
            if col in X_cf.columns:
                X_cf[col] = val
        cf_prob = predict_prob_single(pipeline, X_cf)
        shift_pct = (base_prob - cf_prob) * 100.0  # + favors P1, - favors P2

        # Enforce directional consistency so tree split noise never credits the inferior stat
        expected = true_sign.get(label, 0)
        if expected == 0 or (shift_pct * expected < 0):
            shift_pct = 0.0

        if abs(shift_pct) < 0.05:
            if expected > 0:
                fav = f"{p1_name} (edge)"
            elif expected < 0:
                fav = f"{p2_name} (edge)"
            else:
                fav = "Even"
        elif shift_pct > 0:
            fav = f"{p1_name} (+{shift_pct:.1f}%)"
        else:
            fav = f"{p2_name} (+{abs(shift_pct):.1f}%)"
        driver_rows.append({
            "Factor": label,
            p1_name: p1_disp,
            p2_name: p2_disp,
            "Model Edge": fav,
            "P1_Shift": round(shift_pct, 2),
            "Abs_Impact": abs(shift_pct),
        })

    # Add historical Head-to-Head & Career Overall Win Rate from matches table
    p1_uid = p1.get("player_id", "")
    p2_uid = p2.get("player_id", "")
    if not p1_hist.empty and p2_uid:
        h2h = p1_hist[p1_hist["Opponent_UID"] == p2_uid]
        h2h_w1 = int(h2h["Won"].sum()) if not h2h.empty else 0
        h2h_w2 = len(h2h) - h2h_w1
    else:
        h2h_w1, h2h_w2 = 0, 0

    h2h_edge = "Even (0–0)" if (h2h_w1 == h2h_w2) else (f"{p1_name} ({h2h_w1}–{h2h_w2})" if h2h_w1 > h2h_w2 else f"{p2_name} ({h2h_w2}–{h2h_w1})")
    driver_rows.append({
        "Factor": "Head-to-Head (2015–Now)",
        p1_name: f"{h2h_w1} wins",
        p2_name: f"{h2h_w2} wins",
        "Model Edge": h2h_edge,
        "P1_Shift": 0.0,
        "Abs_Impact": -1.0,
    })

    w1_tot = int(p1_hist["Won"].sum()) if not p1_hist.empty else 0
    n1_tot = len(p1_hist)
    w2_tot = int(p2_hist["Won"].sum()) if not p2_hist.empty else 0
    n2_tot = len(p2_hist)
    pct1 = (w1_tot / n1_tot * 100.0) if n1_tot else 0.0
    pct2 = (w2_tot / n2_tot * 100.0) if n2_tot else 0.0
    win_edge = "Even" if abs(pct1 - pct2) < 0.5 else (f"{p1_name} ({pct1:.1f}%)" if pct1 > pct2 else f"{p2_name} ({pct2:.1f}%)")
    driver_rows.append({
        "Factor": "Overall Career Wins (2015–Now)",
        p1_name: f"{w1_tot}W–{n1_tot - w1_tot}L ({pct1:.1f}%)",
        p2_name: f"{w2_tot}W–{n2_tot - w2_tot}L ({pct2:.1f}%)",
        "Model Edge": win_edge,
        "P1_Shift": 0.0,
        "Abs_Impact": -2.0,
    })

    return pd.DataFrame(driver_rows)


def render_match_drivers(pipeline, X_row, p1, p2, p1_name: str, p2_name: str, surface: str):
    p1_hist = load_player_matches(p1.get("player_id", ""))
    p2_hist = load_player_matches(p2.get("player_id", ""))
    drv_df = explain_matchup_drivers(pipeline, X_row, p1, p2, p1_name, p2_name, surface, p1_hist, p2_hist)

    model_only = drv_df[drv_df["Abs_Impact"] >= 0].sort_values("Abs_Impact", ascending=False).reset_index(drop=True)
    top3 = model_only.head(3)
    reasons = [f"**{r['Factor']}** ({r['Model Edge']})" for _, r in top3.iterrows() if r["Model Edge"] != "Even"]
    if reasons:
        st.info("🧠 **Principal Drivers Behind This Prediction:** " + " • ".join(reasons))

    c_left, c_right = st.columns([1.3, 1])
    with c_left:
        st.dataframe(drv_df[["Factor", p1_name, p2_name, "Model Edge"]], use_container_width=True, hide_index=True)
    with c_right:
        chart_df = model_only.copy()
        chart_df["Favors"] = np.where(chart_df["P1_Shift"] >= 0, p1_name, p2_name)
        fig = px.bar(
            chart_df,
            x="P1_Shift",
            y="Factor",
            color="Favors",
            orientation="h",
            title=f"Win Probability Push (+ favors {p1_name}, − favors {p2_name})",
            labels={"P1_Shift": "Win Probability Impact (%)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


def build_matchup_row(p1, p2, tour: str, surface: str, best_of: int, feature_cols: list) -> pd.DataFrame:
    surf_col = f"elo_{surface.lower()}"
    p1_elo = float(p1["elo_overall"])
    p2_elo = float(p2["elo_overall"])
    p1_surf = float(p1[surf_col])
    p2_surf = float(p2[surf_col])
    p1_age = float(p1.get("age", 26.0))
    p2_age = float(p2.get("age", 26.0))
    p1_pts = float(p1.get("ranking_points", 0))
    p2_pts = float(p2.get("ranking_points", 0))
    p1_slams = float(p1.get("grand_slams", 0))
    p2_slams = float(p2.get("grand_slams", 0))
    p1_mast = float(p1.get("masters_1000", 0))
    p2_mast = float(p2.get("masters_1000", 0))
    p1_tit = float(p1.get("total_titles", 0))
    p2_tit = float(p2.get("total_titles", 0))
    p1_f10 = float(p1.get("form_10", 0.5))
    p2_f10 = float(p2.get("form_10", 0.5))
    p1_m14 = float(p1.get("matches_14d", 0))
    p2_m14 = float(p2.get("matches_14d", 0))

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
        "age_diff": p1_age - p2_age,
        "p1_age": p1_age,
        "p2_age": p2_age,
        "p1_peak_dist": abs(p1_age - 26.0),
        "p2_peak_dist": abs(p2_age - 26.0),
        "pts_diff": p1_pts - p2_pts,
        "log_pts_ratio": math.log((p1_pts + 100.0) / (p2_pts + 100.0)),
        "slam_diff": p1_slams - p2_slams,
        "masters_diff": p1_mast - p2_mast,
        "titles_diff": p1_tit - p2_tit,
        "form10_diff": p1_f10 - p2_f10,
        "fatigue14_diff": p1_m14 - p2_m14,
        "is_atp": 1 if tour == "ATP" else 0,
        "best_of": best_of,
        "surf_hard": 1 if surface == "Hard" else 0,
        "surf_clay": 1 if surface == "Clay" else 0,
        "surf_grass": 1 if surface == "Grass" else 0,
        "p1_lefty": 1 if p1.get("hand") == "L" else 0,
        "p2_lefty": 1 if p2.get("hand") == "L" else 0,
    }
    return pd.DataFrame([row])[feature_cols]


def build_player_lookup(players_df: pd.DataFrame) -> dict:
    lookup = {}
    for row in players_df.to_dict(orient="records"):
        key = (row["tour"], _norm_name(row["player_name"]))
        if key not in lookup:
            lookup[key] = row
    return lookup


def get_player_or_default(lookup: dict, tour: str, name: str) -> dict:
    return lookup.get(
        (tour, _norm_name(name)),
        {
            "player_id": "",
            "tour": tour,
            "player_name": name,
            "hand": "R",
            "matches_played": 0,
            "elo_overall": 1500.0,
            "elo_hard": 1500.0,
            "elo_clay": 1500.0,
            "elo_grass": 1500.0,
            "age": 26.0,
            "ranking_points": 0,
            "grand_slams": 0,
            "masters_1000": 0,
            "total_titles": 0,
            "form_10": 0.5,
            "matches_14d": 0,
        },
    )


st.title("🎾 ATP & WTA Machine Learning Match Predictor Lab")
st.caption("Chronological Surface-Elo + Age + Ranking Points + Career Trophies Engine • Live Supabase & ESPN Sync")

players_df = load_players()
player_lookup = build_player_lookup(players_df)
leaderboard_df = load_leaderboard()
feature_cols = load_feature_cols()
pipelines = load_all_pipelines(leaderboard_df["full_label"].tolist())

tab_today, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Today's Match Forecasts",
    "🔮 Custom Matchup Simulator",
    "🏆 Model Leaderboard & Overfitting Lab",
    "📈 Diagnostic Graphs & Learning Curves",
    "📊 ATP / WTA Elo & Trophy Rankings",
    "📋 Player Match History",
])

with tab_today:
    st.subheader("Live & Upcoming ATP / WTA Match Forecasts (Next 48 Hours)")
    fc1, fc2, fc3 = st.columns([1, 2, 1])
    slate_tour = fc1.selectbox("Tour Filter", ["ALL", "ATP", "WTA"], key="slate_tour")
    slate_model = fc2.selectbox("Primary Forecast Model", leaderboard_df["full_label"].tolist(), index=0, key="slate_model")
    min_conf = fc3.slider("Min Win Probability (%)", 50, 90, 50, key="slate_conf")

    upcoming_df = fetch_upcoming_espn_matches()
    if upcoming_df.empty:
        st.info(
            "No scheduled ATP or WTA singles matches were found on ESPN for the next 48 hours "
            "(either matches are between tournaments or in the off-season). Use **🔮 Custom Matchup Simulator** to simulate any pairing!"
        )
    else:
        if slate_tour != "ALL":
            upcoming_df = upcoming_df[upcoming_df["Tour"] == slate_tour].reset_index(drop=True)

        forecast_rows = []
        detailed_matches = []
        for _, m in upcoming_df.iterrows():
            p1 = get_player_or_default(player_lookup, m["Tour"], m["Player1"])
            p2 = get_player_or_default(player_lookup, m["Tour"], m["Player2"])
            X_m = build_matchup_row(p1, p2, m["Tour"], m["Surface"], int(m["BestOf"]), feature_cols)
            prob1_primary = predict_prob_single(pipelines[slate_model], X_m)
            prob2_primary = 1.0 - prob1_primary
            all_p1_probs = [predict_prob_single(pipelines[lbl], X_m) for lbl in leaderboard_df["full_label"]]
            p1_votes = sum(1 for pr in all_p1_probs if pr >= 0.5)
            total_models = len(all_p1_probs)

            if prob1_primary >= 0.5:
                fav_name, fav_prob, consensus = m["Player1"], prob1_primary * 100, f"{p1_votes}/{total_models} models"
            else:
                fav_name, fav_prob, consensus = m["Player2"], prob2_primary * 100, f"{total_models - p1_votes}/{total_models} models"

            if fav_prob < min_conf:
                continue

            match_label = f"[{m['Tour']}] {m['Player1']} vs {m['Player2']} — {m['Tournament']}"
            forecast_rows.append({
                "Time / Status": m["Status"],
                "Tour": m["Tour"],
                "Tournament": m["Tournament"],
                "Surface": m["Surface"],
                "Matchup": f"{m['Player1']} vs {m['Player2']}",
                "Predicted Winner": f"🏆 {fav_name}",
                "Win Prob %": round(fav_prob, 1),
                "Model Consensus": consensus,
                "P1 Elo / Pts": f"{p1['elo_overall']:.0f} / {int(p1['ranking_points']):,}",
                "P2 Elo / Pts": f"{p2['elo_overall']:.0f} / {int(p2['ranking_points']):,}",
            })
            detailed_matches.append((match_label, m, p1, p2, X_m, prob1_primary))

        if not forecast_rows:
            st.warning("No scheduled matches match your current filter settings.")
        else:
            f_df = pd.DataFrame(forecast_rows)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Matches on Slate", len(f_df))
            k2.metric("ATP Matches", int((f_df["Tour"] == "ATP").sum()))
            k3.metric("WTA Matches", int((f_df["Tour"] == "WTA").sum()))
            k4.metric("Strong Favorites (≥65%)", int((f_df["Win Prob %"] >= 65.0).sum()))
            st.dataframe(f_df.style.format({"Win Prob %": "{:.1f}%"}), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Inspect Any Scheduled Match in Detail")
            selected_idx = st.selectbox("Choose a match from today's slate", range(len(detailed_matches)), format_func=lambda i: detailed_matches[i][0])
            _, sel_m, sel_p1, sel_p2, sel_X, sel_prob1 = detailed_matches[selected_idx]
            s_col = f"elo_{sel_m['Surface'].lower()}"

            dm1, dm2, dm3 = st.columns([2, 1, 2])
            with dm1:
                st.metric(
                    label=f"{sel_m['Player1']} (Age {sel_p1['age']:.1f} • {int(sel_p1['ranking_points']):,} pts • {int(sel_p1['grand_slams'])} GS / {int(sel_p1['masters_1000'])} M1000)",
                    value=f"{sel_prob1:.1%} Win Chance",
                    delta=f"Elo {sel_p1['elo_overall']:.0f} ({sel_m['Surface']}: {sel_p1[s_col]:.0f})",
                )
            with dm2:
                st.markdown(f"<div style='text-align:center; padding-top:10px;'><b>{sel_m['Surface']}</b><br>{sel_m['Status']}</div>", unsafe_allow_html=True)
            with dm3:
                st.metric(
                    label=f"{sel_m['Player2']} (Age {sel_p2['age']:.1f} • {int(sel_p2['ranking_points']):,} pts • {int(sel_p2['grand_slams'])} GS / {int(sel_p2['masters_1000'])} M1000)",
                    value=f"{(1.0 - sel_prob1):.1%} Win Chance",
                    delta=f"Elo {sel_p2['elo_overall']:.0f} ({sel_m['Surface']}: {sel_p2[s_col]:.0f})",
                )

            st.subheader("Why the Model Favors This Player (Key Variables)")
            render_match_drivers(pipelines[slate_model], sel_X, sel_p1, sel_p2, sel_m["Player1"], sel_m["Player2"], sel_m["Surface"])

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
                label=f"{p1_name} (Age {p1_row['age']:.1f} • {int(p1_row['ranking_points']):,} pts)",
                value=f"{p1_prob:.1%} Win Chance",
                delta=f"Elo {p1_row['elo_overall']:.0f} • {int(p1_row['grand_slams'])} GS / {int(p1_row['masters_1000'])} M1000",
            )
        with m2:
            st.markdown("<h3 style='text-align: center; padding-top: 15px;'>VS</h3>", unsafe_allow_html=True)
        with m3:
            st.metric(
                label=f"{p2_name} (Age {p2_row['age']:.1f} • {int(p2_row['ranking_points']):,} pts)",
                value=f"{p2_prob:.1%} Win Chance",
                delta=f"Elo {p2_row['elo_overall']:.0f} • {int(p2_row['grand_slams'])} GS / {int(p2_row['masters_1000'])} M1000",
            )

        st.subheader("Why the Model Favors This Player (Key Variables)")
        render_match_drivers(pipelines[primary_model], X_match, p1_row, p2_row, p1_name, p2_name, surface)

        p1_hist = load_player_matches(p1_row["player_id"])
        p2_hist = load_player_matches(p2_row["player_id"])

        if not p1_hist.empty and not p2_hist.empty:
            h2h = p1_hist[p1_hist["Opponent_UID"] == p2_row["player_id"]]
            p1_h2h_wins = int(h2h["Won"].sum()) if not h2h.empty else 0
            p2_h2h_wins = len(h2h) - p1_h2h_wins
            st.subheader(f"Recent Form & Head-to-Head ({p1_name} {p1_h2h_wins} – {p2_h2h_wins} {p2_name})")
            if not h2h.empty:
                with st.expander(f"View All {len(h2h)} Head-to-Head Meetings (2015–Present)"):
                    st.dataframe(h2h[["Date", "Result", "Score", "Tournament", "Surface", "Round"]], use_container_width=True, hide_index=True)

            rc1, rc2 = st.columns(2)
            with rc1:
                last10_p1 = p1_hist.head(10)
                w10_1 = int(last10_p1["Won"].sum())
                st.markdown(f"**{p1_name} — Latest 10 Matches ({w10_1}W – {len(last10_p1) - w10_1}L)**")
                st.dataframe(last10_p1[["Date", "Result", "Opponent", "Score", "Surface", "Tournament"]], use_container_width=True, hide_index=True)
            with rc2:
                last10_p2 = p2_hist.head(10)
                w10_2 = int(last10_p2["Won"].sum())
                st.markdown(f"**{p2_name} — Latest 10 Matches ({w10_2}W – {len(last10_p2) - w10_2}L)**")
                st.dataframe(last10_p2[["Date", "Result", "Opponent", "Score", "Surface", "Tournament"]], use_container_width=True, hide_index=True)

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
            votes_df, x=f"{p1_name} Win %", y="Model", color="Feature Mode",
            orientation="h", range_x=[0, 100], text=f"{p1_name} Win %",
            title=f"Predicted Win Probability for {p1_name} Across Every Model",
        )
        fig_votes.add_vline(x=50, line_dash="dash", line_color="gray")
        fig_votes.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
        st.plotly_chart(fig_votes, use_container_width=True)

        with st.expander("Inspect Exact Matchup Feature Vector Fed into Models"):
            st.dataframe(X_match, use_container_width=True)

with tab2:
    st.subheader("Train (2016–2024) vs. Out-of-Sample Test (2025–2026) Benchmark")
    def _classify_fit(row):
        if row["overfit_gap"] > 0.05:
            return "⚠️ Overfitted"
        if row["test_acc"] < 0.65:
            return "📉 Underfitted"
        return "✅ Good Fit"

    lb_with_fit = leaderboard_df.copy()
    lb_with_fit.insert(3, "Fit Status", lb_with_fit.apply(_classify_fit, axis=1))

    fc_m, fc_s = st.columns(2)
    mode_filter = fc_m.multiselect("Filter by Feature Pipeline", options=lb_with_fit["feature_mode"].unique().tolist(), default=lb_with_fit["feature_mode"].unique().tolist())
    fit_filter = fc_s.multiselect("Filter by Fit Status", options=lb_with_fit["Fit Status"].unique().tolist(), default=lb_with_fit["Fit Status"].unique().tolist())
    filtered_lb = lb_with_fit[lb_with_fit["feature_mode"].isin(mode_filter) & lb_with_fit["Fit Status"].isin(fit_filter)].copy()

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(name="Train Accuracy (2016-2024)", x=filtered_lb["full_label"], y=filtered_lb["train_acc"] * 100, marker_color="#4C72B0"))
    fig_gap.add_trace(go.Bar(name="Out-of-Sample Test Accuracy (2025-2026)", x=filtered_lb["full_label"], y=filtered_lb["test_acc"] * 100, marker_color="#55A868"))
    fig_gap.update_layout(barmode="group", title="Train vs. Test Accuracy", yaxis_title="Accuracy (%)", yaxis_range=[58, 75], height=430)
    st.plotly_chart(fig_gap, use_container_width=True)

    st.dataframe(
        filtered_lb.style.format({
            "train_acc": "{:.2%}", "test_acc": "{:.2%}", "overfit_gap": "{:+.2%}",
            "train_auc": "{:.4f}", "test_auc": "{:.4f}", "train_brier": "{:.4f}",
            "test_brier": "{:.4f}", "train_seconds": "{:.2f}s",
        }),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.subheader("Model Diagnostics: Learning Curves, ROC Curves & PCA Variance")
    plot_specs = [
        ("models/plots/5_learning_curve.png", "1. Training vs. Validation Learning Curve", "Shows how model accuracy scales as more historical matches are added."),
        ("models/plots/3_roc_curves.png", "2. Out-of-Sample ROC Curves", "Compares True Positive Rate vs. False Positive Rate across thresholds."),
        ("models/plots/2_pca_vs_raw_auc.png", "3. Raw Features vs. PCA (95% Variance) Impact", "Compares each algorithm before and after PCA."),
        ("models/plots/4_pca_variance.png", "4. PCA Scree & Cumulative Explained Variance Plot", "Shows explained variance per principal component."),
        ("models/plots/1_train_vs_test_overfit.png", "5. Train vs. Test Overfitting Gap", "Highlights generalization vs. overfitting."),
    ]
    for img_path, title, desc in plot_specs:
        if os.path.exists(img_path):
            st.markdown(f"### {title}")
            st.write(desc)
            st.image(img_path, use_container_width=True)
            st.markdown("---")

with tab4:
    st.subheader("Live Supabase Player Elo, Age, Points & Trophy Rankings")
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
        view_df[["player_name", "tour", "age", "ranking_points", "grand_slams", "masters_1000", "total_titles", "elo_overall", "elo_hard", "elo_clay", "elo_grass"]],
        use_container_width=True,
    )

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
        total_w = int(match_log["Won"].sum())
        total_l = len(match_log) - total_w
        win_pct = (total_w / len(match_log)) * 100 if len(match_log) else 0.0
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Career Record (2015–Now)", f"{total_w}W – {total_l}L", f"{win_pct:.1f}% Win Rate")
        k2.metric("Age & Official Points", f"{sel_player['age']:.1f} yrs", f"{int(sel_player['ranking_points']):,} pts")
        k3.metric("Grand Slams / Masters 1000", f"{int(sel_player['grand_slams'])} GS / {int(sel_player['masters_1000'])} M1000", f"{int(sel_player['total_titles'])} Total Titles")
        k4.metric("Current Overall Elo", f"{sel_player['elo_overall']:.0f}", f"H:{sel_player['elo_hard']:.0f} C:{sel_player['elo_clay']:.0f} G:{sel_player['elo_grass']:.0f}")

        elo_series = match_log.dropna(subset=["Player Elo"]).iloc[::-1].copy()
        if len(elo_series) > 1:
            fig_elo = px.line(elo_series, x="Date", y="Player Elo", hover_data=["Result", "Opponent", "Score", "Tournament", "Surface"], title=f"{hist_player_name} — Pre-Match Elo Rating Over Time")
            fig_elo.update_traces(line_color="#1f77b4", line_width=2)
            fig_elo.update_layout(height=320, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig_elo, use_container_width=True)

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
            filtered_log = filtered_log[filtered_log["Opponent"].str.contains(q, case=False, na=False) | filtered_log["Tournament"].str.contains(q, case=False, na=False)]

        if hist_limit == "Latest 15":
            filtered_log = filtered_log.head(15)
        elif hist_limit == "Latest 50":
            filtered_log = filtered_log.head(50)

        st.caption(f"Showing {len(filtered_log)} matches")
        st.dataframe(filtered_log[["Date", "Result", "Opponent", "Score", "Tournament", "Surface", "Round", "Player Elo", "Opp Elo"]], use_container_width=True, hide_index=True)


