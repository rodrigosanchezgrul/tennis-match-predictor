import io
import json
import math
import os
import re
import time
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
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

CURRENT_YEAR = datetime.now(timezone.utc).year
CUTOFF_DATE = "2025-01-01"


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def fetch_all_table(table_name: str, order_col: str = None) -> pd.DataFrame:
    rows = []
    start = 0
    step = 1000
    while True:
        q = sb.table(table_name).select("*").range(start, start + step - 1)
        if order_col:
            q = q.order(order_col, desc=False)
        res = q.execute()
        if not res.data:
            break
        rows.extend(res.data)
        if len(res.data) < step:
            break
        start += step
    return pd.DataFrame(rows)


def fetch_sackmann_archive_and_stats():
    """Fetches birthdates, pre-2015 trophies, and 2015+ match serve/return box-score stats."""
    meta = {}
    match_stats = {}

    for tour, repo, prefix in [
        ("ATP", "tennis_atp", "atp"),
        ("WTA", "tennis_wta", "wta"),
    ]:
        p_url = f"https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{prefix}_players.csv"
        try:
            req = urllib.request.Request(p_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                df_p = pd.read_csv(io.BytesIO(resp.read()), low_memory=False)
            for _, r in df_p.iterrows():
                full = f"{r.get('name_first', '')} {r.get('name_last', '')}".strip()
                k = (tour, norm_name(full))
                dob_raw = str(r.get("dob", "")).split(".")[0]
                bdate = None
                if len(dob_raw) == 8 and dob_raw.isdigit():
                    try:
                        bdate = f"{dob_raw[:4]}-{dob_raw[4:6]}-{dob_raw[6:8]}"
                    except Exception:
                        bdate = None
                if k not in meta:
                    meta[k] = {"birth_date": bdate, "slams": 0, "masters": 0, "titles": 0}
                elif bdate and not meta[k]["birth_date"]:
                    meta[k]["birth_date"] = bdate
        except Exception as e:
            print(f"Warning fetching {prefix}_players.csv: {e}")

        for yr in range(1995, CURRENT_YEAR + 1):
            m_url = f"https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{prefix}_matches_{yr}.csv"
            try:
                req = urllib.request.Request(m_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    df_m = pd.read_csv(io.BytesIO(resp.read()), low_memory=False)
            except Exception:
                continue

            if yr < 2015:
                finals = df_m[df_m["round"] == "F"]
                for _, r in finals.iterrows():
                    w_name = str(r.get("winner_name", ""))
                    lvl = str(r.get("tourney_level", ""))
                    k = (tour, norm_name(w_name))
                    if k not in meta:
                        meta[k] = {"birth_date": None, "slams": 0, "masters": 0, "titles": 0}
                    meta[k]["titles"] += 1
                    if lvl == "G":
                        meta[k]["slams"] += 1
                    elif lvl in ("M", "PM", "P"):
                        meta[k]["masters"] += 1
            else:
                for _, r in df_m.iterrows():
                    t_date = str(r.get("tourney_date", "")).split(".")[0]
                    if len(t_date) == 8:
                        iso_d = f"{t_date[:4]}-{t_date[4:6]}-{t_date[6:8]}"
                    else:
                        continue
                    w_n = norm_name(str(r.get("winner_name", "")))
                    l_n = norm_name(str(r.get("loser_name", "")))
                    if not w_n or not l_n:
                        continue
                    match_stats[(tour, iso_d, w_n, l_n)] = {
                        "w_svpt": r.get("w_svpt"), "w_1stWon": r.get("w_1stWon"), "w_2ndWon": r.get("w_2ndWon"),
                        "w_bpSaved": r.get("w_bpSaved"), "w_bpFaced": r.get("w_bpFaced"),
                        "l_svpt": r.get("l_svpt"), "l_1stWon": r.get("l_1stWon"), "l_2ndWon": r.get("l_2ndWon"),
                        "l_bpSaved": r.get("l_bpSaved"), "l_bpFaced": r.get("l_bpFaced"),
                    }

    return meta, match_stats


def fetch_espn_live_points():
    pts_map = {}
    for slug, tour in [("atp", "ATP"), ("wta", "WTA")]:
        url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/{slug}/rankings?limit=300"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for ranking_block in data.get("rankings", []):
                for entry in ranking_block.get("ranks", []):
                    ath = entry.get("athlete", {})
                    name = ath.get("displayName") or ath.get("fullName") or ""
                    pts = entry.get("points")
                    if name and pts is not None:
                        pts_map[(tour, norm_name(name))] = int(float(pts))
        except Exception as e:
            print(f"Warning fetching ESPN {tour} rankings: {e}")
    return pts_map


def infer_level(tourney_name: str, existing_level: str) -> str:
    if existing_level in ("G", "M", "PM", "P", "A", "F"):
        return existing_level
    n = (tourney_name or "").lower()
    if any(g in n for g in ["australian open", "roland garros", "french open", "wimbledon", "us open"]):
        return "G"
    if any(m in n for m in ["masters", "1000", "indian wells", "miami", "madrid", "rome", "canada", "cincinnati", "shanghai", "paris"]):
        return "M"
    return "A"


def is_straight_sets(score_str: str) -> float:
    s = str(score_str or "").strip()
    if not s or "RET" in s or "W/O" in s:
        return 0.5
    sets = [tok for tok in s.split() if "-" in tok]
    if len(sets) == 2:
        return 1.0
    if len(sets) >= 3:
        return 0.0
    return 0.5


def build_training_dataset(players_df: pd.DataFrame, matches_df: pd.DataFrame, meta: dict, match_stats: dict, live_pts: dict):
    p_state = {}
    for _, r in players_df.iterrows():
        uid = r["player_id"]
        tour = r["tour"]
        name = r["player_name"]
        k = (tour, norm_name(name))
        s = meta.get(k, {"birth_date": None, "slams": 0, "masters": 0, "titles": 0})
        raw_bd = r.get("birth_date")
        bdate_str = None if (raw_bd is None or pd.isna(raw_bd)) else str(raw_bd)
        if not bdate_str:
            bdate_str = s.get("birth_date")
        bdate = datetime.strptime(str(bdate_str)[:10], "%Y-%m-%d").date() if (bdate_str and str(bdate_str).lower() != "nan") else None

        p_state[uid] = {
            "tour": tour,
            "player_name": name,
            "hand": r.get("hand") or "R",
            "birth_date": bdate,
            "slams": s["slams"],
            "masters": s["masters"],
            "titles": s["titles"],
            "recent_results": deque(maxlen=10),
            "recent_dates": deque(),
            "pts_events": deque(),
            "srv_hist": deque(maxlen=20),
            "ret_hist": deque(maxlen=20),
            "bp_hist": deque(maxlen=20),
            "straight_hist": deque(maxlen=20),
            "last_match_date": None,
        }

    matches_df = matches_df.sort_values("tourney_date", ascending=True).reset_index(drop=True)
    rows = []

    for _, m in matches_df.iterrows():
        w_uid = m.get("winner_uid")
        l_uid = m.get("loser_uid")
        if w_uid not in p_state or l_uid not in p_state:
            continue

        m_date_str = str(m.get("tourney_date", ""))[:10]
        try:
            m_date = datetime.strptime(m_date_str, "%Y-%m-%d").date()
        except Exception:
            continue

        tour = m.get("tour") or p_state[w_uid]["tour"]
        surface = m.get("surface") or "Hard"
        if surface not in ("Hard", "Clay", "Grass"):
            surface = "Hard"
        best_of = int(m.get("best_of") or 3)
        rnd = str(m.get("round") or "")
        t_name = str(m.get("tourney_name") or "")
        lvl = infer_level(t_name, str(m.get("tourney_level") or ""))

        w = p_state[w_uid]
        l = p_state[l_uid]

        for st in (w, l):
            while st["recent_dates"] and (m_date - st["recent_dates"][0]).days > 14:
                st["recent_dates"].popleft()
            while st["pts_events"] and (m_date - st["pts_events"][0][0]).days > 365:
                st["pts_events"].popleft()

        def get_snap(st, elo_ov, elo_sf):
            age = ((m_date - st["birth_date"]).days / 365.25) if st["birth_date"] else 26.0
            pts = sum(x[1] for x in st["pts_events"])
            f10 = (sum(st["recent_results"]) / len(st["recent_results"])) if st["recent_results"] else 0.5
            m14 = len(st["recent_dates"])
            srv = float(np.mean(st["srv_hist"])) if st["srv_hist"] else 0.62
            ret = float(np.mean(st["ret_hist"])) if st["ret_hist"] else 0.38
            bps = float(np.mean(st["bp_hist"])) if st["bp_hist"] else 0.60
            dom = ret / max(1.0 - srv, 0.15)
            ss = float(np.mean(st["straight_hist"])) if st["straight_hist"] else 0.50
            inact = min((m_date - st["last_match_date"]).days, 180) if st["last_match_date"] else 14
            return {
                "elo": float(elo_ov), "surf_elo": float(elo_sf),
                "age": age, "peak_dist": abs(age - 26.0),
                "pts": pts, "slams": st["slams"], "masters": st["masters"], "titles": st["titles"],
                "form10": f10, "m14": m14, "lefty": 1 if st["hand"] == "L" else 0,
                "srv": srv, "ret": ret, "bps": bps, "dom": dom, "ss": ss, "inact": inact,
            }

        w_elo = float(m.get("w_pre_elo") or 1500.0)
        l_elo = float(m.get("l_pre_elo") or 1500.0)
        w_sf = float(m.get("w_pre_surf_elo") or w_elo)
        l_sf = float(m.get("l_pre_surf_elo") or l_elo)

        ws = get_snap(w, w_elo, w_sf)
        ls = get_snap(l, l_elo, l_sf)

        if m_date_str >= "2016-01-01":
            for target, p1, p2 in [(1, ws, ls), (0, ls, ws)]:
                elo_prob = 1.0 / (1.0 + 10.0 ** ((p2["elo"] - p1["elo"]) / 400.0))
                surf_prob = 1.0 / (1.0 + 10.0 ** ((p2["surf_elo"] - p1["surf_elo"]) / 400.0))
                rows.append({
                    "tourney_date": m_date_str,
                    "target": target,
                    "elo_diff": p1["elo"] - p2["elo"],
                    "surf_elo_diff": p1["surf_elo"] - p2["surf_elo"],
                    "elo_win_prob": elo_prob,
                    "surf_elo_win_prob": surf_prob,
                    "p1_elo": p1["elo"], "p2_elo": p2["elo"],
                    "p1_surf_elo": p1["surf_elo"], "p2_surf_elo": p2["surf_elo"],
                    "age_diff": p1["age"] - p2["age"],
                    "p1_age": p1["age"], "p2_age": p2["age"],
                    "p1_peak_dist": p1["peak_dist"], "p2_peak_dist": p2["peak_dist"],
                    "pts_diff": p1["pts"] - p2["pts"],
                    "log_pts_ratio": math.log((p1["pts"] + 100.0) / (p2["pts"] + 100.0)),
                    "slam_diff": p1["slams"] - p2["slams"],
                    "masters_diff": p1["masters"] - p2["masters"],
                    "titles_diff": p1["titles"] - p2["titles"],
                    "form10_diff": p1["form10"] - p2["form10"],
                    "fatigue14_diff": p1["m14"] - p2["m14"],
                    "srv_diff": p1["srv"] - p2["srv"],
                    "ret_diff": p1["ret"] - p2["ret"],
                    "bp_saved_diff": p1["bps"] - p2["bps"],
                    "dom_ratio_diff": p1["dom"] - p2["dom"],
                    "straight_set_diff": p1["ss"] - p2["ss"],
                    "inactivity_diff": p1["inact"] - p2["inact"],
                    "is_atp": 1 if tour == "ATP" else 0,
                    "best_of": best_of,
                    "surf_hard": 1 if surface == "Hard" else 0,
                    "surf_clay": 1 if surface == "Clay" else 0,
                    "surf_grass": 1 if surface == "Grass" else 0,
                    "p1_lefty": p1["lefty"],
                    "p2_lefty": p2["lefty"],
                })

        # Post-match state updates
        w["recent_results"].append(1)
        l["recent_results"].append(0)
        w["recent_dates"].append(m_date)
        l["recent_dates"].append(m_date)
        w["last_match_date"] = m_date
        l["last_match_date"] = m_date

        ss_val = is_straight_sets(m.get("score"))
        w["straight_hist"].append(ss_val)
        l["straight_hist"].append(0.0)

        # Update serve/return rolling stats if box score exists
        box = match_stats.get((tour, m_date_str, norm_name(w["player_name"]), norm_name(l["player_name"])))
        if box:
            try:
                w_sv, l_sv = float(box["w_svpt"]), float(box["l_svpt"])
                if w_sv > 0 and l_sv > 0:
                    w_sw = (float(box["w_1stWon"]) + float(box["w_2ndWon"])) / w_sv
                    l_sw = (float(box["l_1stWon"]) + float(box["l_2ndWon"])) / l_sv
                    w["srv_hist"].append(w_sw)
                    l["srv_hist"].append(l_sw)
                    w["ret_hist"].append(1.0 - l_sw)
                    l["ret_hist"].append(1.0 - w_sw)
                if float(box["w_bpFaced"] or 0) > 0:
                    w["bp_hist"].append(float(box["w_bpSaved"]) / float(box["w_bpFaced"]))
                if float(box["l_bpFaced"] or 0) > 0:
                    l["bp_hist"].append(float(box["l_bpSaved"]) / float(box["l_bpFaced"]))
            except Exception:
                pass

        base_pts = 180 if lvl == "G" else (90 if lvl in ("M", "PM", "P") else 35)
        if rnd == "F":
            w["titles"] += 1
            if lvl == "G":
                w["slams"] += 1
            elif lvl in ("M", "PM", "P"):
                w["masters"] += 1
            w["pts_events"].append((m_date, base_pts * 3))
            l["pts_events"].append((m_date, base_pts * 2))
        else:
            w["pts_events"].append((m_date, base_pts))

    today = datetime.now(timezone.utc).date()
    updates = []
    for uid, p in p_state.items():
        while p["recent_dates"] and (today - p["recent_dates"][0]).days > 14:
            p["recent_dates"].popleft()
        while p["pts_events"] and (today - p["pts_events"][0][0]).days > 365:
            p["pts_events"].popleft()

        espn_pts = live_pts.get((p["tour"], norm_name(p["player_name"])))
        rolling_pts = sum(x[1] for x in p["pts_events"])
        srv = float(np.mean(p["srv_hist"])) if p["srv_hist"] else 0.62
        ret = float(np.mean(p["ret_hist"])) if p["ret_hist"] else 0.38
        bps = float(np.mean(p["bp_hist"])) if p["bp_hist"] else 0.60
        dom = ret / max(1.0 - srv, 0.15)
        ss = float(np.mean(p["straight_hist"])) if p["straight_hist"] else 0.50
        inact = min((today - p["last_match_date"]).days, 180) if p["last_match_date"] else 14

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
            "srv_won_20": round(srv, 4),
            "ret_won_20": round(ret, 4),
            "bp_saved_20": round(bps, 4),
            "dom_ratio_20": round(dom, 4),
            "straight_set_20": round(ss, 4),
            "days_inactive": int(inact),
        })

    for i in range(0, len(updates), 500):
        sb.table("players").upsert(updates[i:i + 500], on_conflict="player_id").execute()

    return pd.DataFrame(rows)


def retrain_models(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("tourney_date", "target")]
    train_df = df[df["tourney_date"] < CUTOFF_DATE]
    test_df = df[df["tourney_date"] >= CUTOFF_DATE]

    X_train, y_train = train_df[feature_cols], train_df["target"].values
    X_test, y_test = test_df[feature_cols], test_df["target"].values

    base_models = {
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000),
        "Calibrated Logistic": CalibratedClassifierCV(LogisticRegression(C=0.5, max_iter=1000), cv=3),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=9, min_samples_leaf=25, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.05, subsample=0.85, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.04, l2_regularization=2.0, random_state=42),
        "Neural Net (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.08, max_iter=300, early_stopping=True, random_state=42),
        "Super-Ensemble (RF+HGB+MLP)": VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=250, max_depth=9, min_samples_leaf=25, n_jobs=-1, random_state=42)),
                ("hgb", HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.04, l2_regularization=2.0, random_state=42)),
                ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.08, max_iter=300, early_stopping=True, random_state=42)),
            ],
            voting="soft",
            weights=[2, 2, 1],
        ),
    }

    os.makedirs("models/saved", exist_ok=True)
    lb_rows = []

    for mode_name, use_pca in [("Raw Features", False), ("PCA (95% Var)", True)]:
        for m_name, clf in base_models.items():
            if use_pca and "Super-Ensemble" in m_name:
                continue
            full_label = f"{m_name} [{mode_name}]"
            steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            if use_pca:
                steps.append(("pca", PCA(n_components=0.95, random_state=42)))
            steps.append(("model", clf))
            pipe = Pipeline(steps)

            t0 = time.time()
            pipe.fit(X_train, y_train)
            elapsed = time.time() - t0

            tr_prob = pipe.predict_proba(X_train)[:, 1]
            te_prob = pipe.predict_proba(X_test)[:, 1]
            tr_acc = accuracy_score(y_train, tr_prob >= 0.5)
            te_acc = accuracy_score(y_test, te_prob >= 0.5)

            lb_rows.append({
                "model_name": m_name,
                "feature_mode": mode_name,
                "full_label": full_label,
                "train_acc": tr_acc,
                "test_acc": te_acc,
                "overfit_gap": tr_acc - te_acc,
                "train_auc": roc_auc_score(y_train, tr_prob),
                "test_auc": roc_auc_score(y_test, te_prob),
                "train_brier": brier_score_loss(y_train, tr_prob),
                "test_brier": brier_score_loss(y_test, te_prob),
                "train_seconds": elapsed,
            })

            safe = (
                full_label.lower()
                .replace(" ", "_").replace("(", "").replace(")", "")
                .replace("[", "").replace("]", "").replace("%", "")
            )
            joblib.dump(pipe, f"models/saved/{safe}.joblib")

    lb_df = pd.DataFrame(lb_rows).sort_values("test_auc", ascending=False).reset_index(drop=True)
    lb_df.to_csv("models/leaderboard.csv", index=False)
    with open("models/feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("\n=== UPGRADED 34-FEATURE + SUPER-ENSEMBLE LEADERBOARD ===")
    print(lb_df[["full_label", "train_acc", "test_acc", "test_auc"]].to_string(index=False))


if __name__ == "__main__":
    print("1/4 Loading players & matches from Supabase...")
    players = fetch_all_table("players")
    matches = fetch_all_table("matches", order_col="tourney_date")
    print("2/4 Fetching birthdates, career trophies, serve/return box scores & live ESPN rankings...")
    meta, match_stats = fetch_sackmann_archive_and_stats()
    live_pts = fetch_espn_live_points()
    print("3/4 Building 34-feature dataset & updating Supabase players...")
    ds = build_training_dataset(players, matches, meta, match_stats, live_pts)
    print(f"4/4 Retraining 13 ML pipelines (including Super-Ensemble) on {len(ds):,} matches...")
    retrain_models(ds)

