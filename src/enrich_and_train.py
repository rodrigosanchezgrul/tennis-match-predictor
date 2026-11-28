import json
import math
import os
import re
import time
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def fetch_all_table(table_name: str, order_col: str = None) -> pd.DataFrame:
    rows = []
    start = 0
    step = 1000
    while True:
        q = sb.table(table_name).select("*")
        if order_col:
            q = q.order(order_col, desc=False)
        res = q.range(start, start + step - 1).execute()
        if not res.data:
            break
        rows.extend(res.data)
        if len(res.data) < step:
            break
        start += step
    return pd.DataFrame(rows)


def fetch_sackmann_seed_metadata() -> dict:
    meta = {}
    base = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
    for tour, prefix in [("ATP", "atp"), ("WTA", "wta")]:
        p_url = f"{base}/{prefix}/{prefix}_players.csv"
        try:
            pdf = pd.read_csv(p_url, low_memory=False)
            for _, r in pdf.iterrows():
                full = f"{r.get('name_first', '')} {r.get('name_last', '')}".strip()
                dob_raw = str(r.get("dob", "")).split(".")[0]
                bdate = None
                if len(dob_raw) == 8 and dob_raw.isdigit():
                    try:
                        bdate = f"{dob_raw[:4]}-{dob_raw[4:6]}-{dob_raw[6:8]}"
                    except Exception:
                        bdate = None
                if full:
                    meta[(tour, norm_name(full))] = {
                        "birth_date": bdate,
                        "pre_slams": 0,
                        "pre_masters": 0,
                        "pre_titles": 0,
                    }
        except Exception as e:
            print(f"[WARN] Could not load {prefix}_players.csv: {e}")

        for yr in range(1995, 2015):
            m_url = f"{base}/{prefix}/{prefix}_matches_{yr}.csv"
            try:
                mdf = pd.read_csv(m_url, usecols=["tourney_level", "round", "winner_name"], low_memory=False)
                finals = mdf[mdf["round"] == "F"]
                for _, fr in finals.iterrows():
                    key = (tour, norm_name(str(fr["winner_name"])))
                    if key not in meta:
                        meta[key] = {"birth_date": None, "pre_slams": 0, "pre_masters": 0, "pre_titles": 0}
                    lvl = str(fr["tourney_level"])
                    meta[key]["pre_titles"] += 1
                    if lvl == "G":
                        meta[key]["pre_slams"] += 1
                    elif lvl in ("M", "PM", "P"):
                        meta[key]["pre_masters"] += 1
            except Exception:
                continue
    return meta


def fetch_espn_live_rankings() -> dict:
    pts_map = {}
    for slug, tour in [("atp", "ATP"), ("wta", "WTA")]:
        url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/{slug}/rankings"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for block in data.get("rankings", []):
                for entry in block.get("ranks", []):
                    ath = entry.get("athlete") or {}
                    name = ath.get("displayName") or ""
                    pts = int(float(entry.get("points") or 0))
                    if name and pts > 0:
                        pts_map[(tour, norm_name(name))] = pts
        except Exception as e:
            print(f"[WARN] ESPN rankings fetch failed for {tour}: {e}")
    return pts_map


def infer_level(tourney_name: str, existing_level: str = "") -> str:
    if existing_level in ("G", "M", "A", "F"):
        return existing_level
    n = (tourney_name or "").lower()
    if any(g in n for g in ["australian open", "roland garros", "french open", "wimbledon", "us open"]):
        return "G"
    if any(m in n for m in ["masters", "1000", "indian wells", "miami", "madrid", "rome", "canada", "cincinnati", "shanghai", "paris", "doha", "dubai", "beijing", "wuhan"]):
        return "M"
    return "A"


def level_points(level: str, round_name: str) -> int:
    base = {"G": 2000, "M": 1000, "F": 1500}.get(level, 500)
    r = (round_name or "").upper()
    mult = {"F": 1.0, "SF": 0.6, "QF": 0.36, "R16": 0.18, "R32": 0.09, "R64": 0.045}.get(r, 0.02)
    return int(base * mult)


def build_training_dataset(players_df: pd.DataFrame, matches_df: pd.DataFrame, seed_meta: dict, live_pts: dict):
    p_info = {}
    for _, r in players_df.iterrows():
        uid = r["player_id"]
        tour = r["tour"]
        key = (tour, norm_name(r["player_name"]))
        s = seed_meta.get(key, {})
        bdate_str = r.get("birth_date") or s.get("birth_date")
        bdate = datetime.strptime(str(bdate_str)[:10], "%Y-%m-%d").date() if bdate_str else None
        p_info[uid] = {
            "player_id": uid,
            "tour": tour,
            "player_name": r["player_name"],
            "hand": r.get("hand") or "R",
            "birth_date": bdate,
            "slams": int(s.get("pre_slams", 0)),
            "masters": int(s.get("pre_masters", 0)),
            "titles": int(s.get("pre_titles", 0)),
            "recent_results": deque(maxlen=10),
            "recent_dates": deque(),
            "pts_events": deque(),
        }

    matches_df = matches_df.sort_values("tourney_date").reset_index(drop=True)
    records = []

    for idx, m in matches_df.iterrows():
        dt_str = str(m["tourney_date"])[:10]
        try:
            m_date = datetime.strptime(dt_str, "%Y-%m-%d").date()
        except Exception:
            continue

        w_uid = m["winner_uid"]
        l_uid = m["loser_uid"]
        if w_uid not in p_info or l_uid not in p_info:
            continue

        pw = p_info[w_uid]
        pl = p_info[l_uid]
        tour = m.get("tour") or pw["tour"]
        surf = m.get("surface") or "Hard"
        best_of = int(m.get("best_of") or 3)
        lvl = infer_level(m.get("tourney_name", ""), str(m.get("tourney_level") or ""))
        rnd = str(m.get("round") or "")

        cutoff_365 = m_date - timedelta(days=365)
        cutoff_14 = m_date - timedelta(days=14)
        for p in (pw, pl):
            while p["pts_events"] and p["pts_events"][0][0] < cutoff_365:
                p["pts_events"].popleft()
            while p["recent_dates"] and p["recent_dates"][0] < cutoff_14:
                p["recent_dates"].popleft()

        def snap(p, elo_all, elo_surf):
            age = ((m_date - p["birth_date"]).days / 365.25) if p["birth_date"] else 26.0
            pts = sum(x[1] for x in p["pts_events"])
            f10 = (sum(p["recent_results"]) / len(p["recent_results"])) if p["recent_results"] else 0.5
            m14 = len(p["recent_dates"])
            return {
                "elo": float(elo_all or 1500.0),
                "surf": float(elo_surf or elo_all or 1500.0),
                "age": age,
                "pts": pts,
                "slams": p["slams"],
                "masters": p["masters"],
                "titles": p["titles"],
                "f10": f10,
                "m14": m14,
                "lefty": 1 if p["hand"] == "L" else 0,
            }

        sw = snap(pw, m.get("w_pre_elo"), m.get("w_pre_surf_elo"))
        sl = snap(pl, m.get("l_pre_elo"), m.get("l_pre_surf_elo"))

        if idx % 2 == 0:
            p1, p2, target = sw, sl, 1
        else:
            p1, p2, target = sl, sw, 0

        elo_prob = 1.0 / (1.0 + 10.0 ** ((p2["elo"] - p1["elo"]) / 400.0))
        surf_prob = 1.0 / (1.0 + 10.0 ** ((p2["surf"] - p1["surf"]) / 400.0))

        records.append({
            "year": m_date.year,
            "target": target,
            "elo_diff": p1["elo"] - p2["elo"],
            "surf_elo_diff": p1["surf"] - p2["surf"],
            "elo_win_prob": elo_prob,
            "surf_elo_win_prob": surf_prob,
            "p1_elo": p1["elo"],
            "p2_elo": p2["elo"],
            "p1_surf_elo": p1["surf"],
            "p2_surf_elo": p2["surf"],
            "age_diff": p1["age"] - p2["age"],
            "p1_age": p1["age"],
            "p2_age": p2["age"],
            "p1_peak_dist": abs(p1["age"] - 26.0),
            "p2_peak_dist": abs(p2["age"] - 26.0),
            "pts_diff": p1["pts"] - p2["pts"],
            "log_pts_ratio": math.log((p1["pts"] + 100.0) / (p2["pts"] + 100.0)),
            "slam_diff": p1["slams"] - p2["slams"],
            "masters_diff": p1["masters"] - p2["masters"],
            "titles_diff": p1["titles"] - p2["titles"],
            "form10_diff": p1["f10"] - p2["f10"],
            "fatigue14_diff": p1["m14"] - p2["m14"],
            "is_atp": 1 if tour == "ATP" else 0,
            "best_of": best_of,
            "surf_hard": 1 if surf == "Hard" else 0,
            "surf_clay": 1 if surf == "Clay" else 0,
            "surf_grass": 1 if surf == "Grass" else 0,
            "p1_lefty": p1["lefty"],
            "p2_lefty": p2["lefty"],
        })

        pw["recent_results"].append(1)
        pl["recent_results"].append(0)
        pw["recent_dates"].append(m_date)
        pl["recent_dates"].append(m_date)
        pw["pts_events"].append((m_date, level_points(lvl, rnd)))
        if rnd == "F":
            pw["titles"] += 1
            if lvl == "G":
                pw["slams"] += 1
            elif lvl == "M":
                pw["masters"] += 1

    today = datetime.now(timezone.utc).date()
    c14 = today - timedelta(days=14)
    c365 = today - timedelta(days=365)
    updates = []
    for uid, p in p_info.items():
        while p["recent_dates"] and p["recent_dates"][0] < c14:
            p["recent_dates"].popleft()
        while p["pts_events"] and p["pts_events"][0][0] < c365:
            p["pts_events"].popleft()
        espn_pts = live_pts.get((p["tour"], norm_name(p["player_name"])))
        rolling_pts = sum(x[1] for x in p["pts_events"])
        orig = players_df[players_df["player_id"] == uid].iloc[0]
        def _num(val, default, cast=float):
            return default if (val is None or pd.isna(val)) else cast(val)
        updates.append({
            "player_id": uid,
            "tour": p["tour"],
            "player_name": p["player_name"],
            "hand": "R" if (p["hand"] is None or pd.isna(p["hand"])) else str(p["hand"]),
            "matches_played": _num(orig.get("matches_played"), 0, int),
            "elo_overall": _num(orig.get("elo_overall"), 1500.0, float),
            "elo_hard": _num(orig.get("elo_hard"), 1500.0, float),
            "elo_clay": _num(orig.get("elo_clay"), 1500.0, float),
            "elo_grass": _num(orig.get("elo_grass"), 1500.0, float),
            "birth_date": p["birth_date"].isoformat() if p["birth_date"] else None,
            "ranking_points": int(espn_pts if espn_pts is not None else rolling_pts),
            "grand_slams": int(p["slams"]),
            "masters_1000": int(p["masters"]),
            "total_titles": int(p["titles"]),
            "form_10": round((sum(p["recent_results"]) / len(p["recent_results"])) if p["recent_results"] else 0.5, 3),
            "matches_14d": int(len(p["recent_dates"])),
        })

    for i in range(0, len(updates), 500):
        sb.table("players").upsert(updates[i:i + 500], on_conflict="player_id").execute()

    return pd.DataFrame(records)


def train_and_save(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("year", "target")]
    os.makedirs("models/saved", exist_ok=True)
    with open("models/feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    train_df = df[(df["year"] >= 2016) & (df["year"] <= 2024)]
    test_df = df[df["year"] >= 2025]
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(max_depth=3, learning_rate=0.05, n_estimators=180, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1),
        "Neural Net (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1.0, max_iter=300, early_stopping=True, random_state=42),
        "Calibrated Logistic": LogisticRegression(max_iter=1000, C=0.1),
    }

    lb_rows = []
    for name, clf in base_models.items():
        for mode in ["Raw Features", "PCA (95% Var)"]:
            full_label = f"{name} [{mode}]"
            t0 = time.time()
            if mode == "Raw Features":
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            else:
                pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.95, random_state=42)), ("clf", clf)])
            pipe.fit(X_train, y_train)
            sec = time.time() - t0

            p_tr = pipe.predict_proba(X_train)[:, 1]
            p_te = pipe.predict_proba(X_test)[:, 1]
            tr_acc = accuracy_score(y_train, p_tr >= 0.5)
            te_acc = accuracy_score(y_test, p_te >= 0.5)

            safe = (
                full_label.lower()
                .replace(" ", "_").replace("(", "").replace(")", "")
                .replace("[", "").replace("]", "").replace("%", "")
            )
            joblib.dump(pipe, f"models/saved/{safe}.joblib")

            lb_rows.append({
                "full_label": full_label,
                "model_name": name,
                "feature_mode": mode,
                "train_acc": round(tr_acc, 4),
                "test_acc": round(te_acc, 4),
                "overfit_gap": round(tr_acc - te_acc, 4),
                "train_auc": round(roc_auc_score(y_train, p_tr), 4),
                "test_auc": round(roc_auc_score(y_test, p_te), 4),
                "train_brier": round(brier_score_loss(y_train, p_tr), 4),
                "test_brier": round(brier_score_loss(y_test, p_te), 4),
                "train_seconds": round(sec, 2),
            })

    lb_df = pd.DataFrame(lb_rows).sort_values("test_auc", ascending=False).reset_index(drop=True)
    lb_df.to_csv("models/leaderboard.csv", index=False)
    print("\n=== NEW OUT-OF-SAMPLE LEADERBOARD (WITH AGE, TROPHIES, POINTS & FORM) ===")
    print(lb_df[["full_label", "train_acc", "test_acc", "test_auc"]].to_string(index=False))


if __name__ == "__main__":
    print("1/4 Loading players & matches from Supabase...")
    players = fetch_all_table("players")
    matches = fetch_all_table("matches", order_col="tourney_date")
    print("2/4 Fetching birthdates & career trophies archive + live ESPN rankings...")
    meta = fetch_sackmann_seed_metadata()
    live_pts = fetch_espn_live_rankings()
    print("3/4 Building chronological feature dataset & updating Supabase players...")
    ds = build_training_dataset(players, matches, meta, live_pts)
    print(f"4/4 Retraining 12 ML pipelines on {len(ds):,} matches...")
    train_and_save(ds)


