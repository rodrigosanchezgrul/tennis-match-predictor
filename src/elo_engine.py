import os
from collections import defaultdict
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from src.espn_fetcher import fetch_espn_matches, _norm_name

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

ARCHIVE_BASE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
YEARS = range(2015, 2027)  # 2015 through 2026


def load_historical_matches() -> pd.DataFrame:
    frames = []
    for tour in ["atp", "wta"]:
        for year in YEARS:
            url = f"{ARCHIVE_BASE}/{tour}/{tour}_matches_{year}.csv"
            try:
                df = pd.read_csv(url, low_memory=False)
                df["tour"] = tour.upper()
                frames.append(df)
                print(f"Loaded {tour.upper()} {year}: {len(df)} matches")
            except Exception as exc:
                print(f"Skipping {tour.upper()} {year} ({exc})")

    matches = pd.concat(frames, ignore_index=True)

    # Map existing (tour, player_name) -> Sackmann player_id so ESPN matches update existing players
    name_to_id = {}
    for row in matches[["tour", "winner_name", "winner_id", "loser_name", "loser_id"]].dropna().itertuples(index=False):
        name_to_id[(row.tour, _norm_name(str(row.winner_name)))] = int(float(row.winner_id))
        name_to_id[(row.tour, _norm_name(str(row.loser_name)))] = int(float(row.loser_id))

    # Append live completed ATP & WTA singles matches from ESPN after the archive's latest date
    archive_dates = pd.to_numeric(matches["tourney_date"], errors="coerce").dropna()
    if not archive_dates.empty:
        max_archive_date = int(archive_dates.max())
        espn_df = fetch_espn_matches(min_date_int=max_archive_date, name_to_id=name_to_id)
        if not espn_df.empty:
            matches = pd.concat([matches, espn_df], ignore_index=True)

    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"].astype(str).str.split(".").str[0], format="%Y%m%d", errors="coerce"
    )
    matches["winner_id"] = pd.to_numeric(matches["winner_id"], errors="coerce")
    matches["loser_id"] = pd.to_numeric(matches["loser_id"], errors="coerce")
    matches = matches.dropna(subset=["tourney_date", "winner_id", "loser_id"])

    # Prefix IDs so ATP and WTA player IDs never collide in a unified table
    matches["winner_uid"] = matches["tour"].str.lower() + "_" + matches["winner_id"].astype(int).astype(str)
    matches["loser_uid"] = matches["tour"].str.lower() + "_" + matches["loser_id"].astype(int).astype(str)

    # Sort strictly chronologically so Elo updates never leak future results
    matches = matches.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)
    return matches


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def k_factor(matches_played: int, tourney_level: str) -> float:
    base_k = 32.0 if matches_played < 30 else 24.0
    if tourney_level == "G":
        return base_k * 1.15
    return base_k


def compute_elo_ratings(matches: pd.DataFrame):
    overall_elo = defaultdict(lambda: 1500.0)
    surface_elo = {
        "Hard": defaultdict(lambda: 1500.0),
        "Clay": defaultdict(lambda: 1500.0),
        "Grass": defaultdict(lambda: 1500.0),
    }
    match_counts = defaultdict(int)
    player_meta = {}

    pre_w_elo, pre_l_elo = [], []
    pre_w_surf, pre_l_surf = [], []

    for row in matches.itertuples(index=False):
        w_id, l_id = row.winner_uid, row.loser_uid
        surf = row.surface if row.surface in surface_elo else None
        level = getattr(row, "tourney_level", "")

        # 1. Record PRE-MATCH ratings (used later as ML training features)
        w_ov = overall_elo[w_id]
        l_ov = overall_elo[l_id]
        pre_w_elo.append(w_ov)
        pre_l_elo.append(l_ov)

        if surf:
            w_sf = surface_elo[surf][w_id]
            l_sf = surface_elo[surf][l_id]
        else:
            w_sf, l_sf = w_ov, l_ov
        pre_w_surf.append(w_sf)
        pre_l_surf.append(l_sf)

        # 2. Update OVERALL Elo
        exp_w = expected_score(w_ov, l_ov)
        k_w = k_factor(match_counts[w_id], level)
        k_l = k_factor(match_counts[l_id], level)
        overall_elo[w_id] = w_ov + k_w * (1.0 - exp_w)
        overall_elo[l_id] = l_ov + k_l * (0.0 - (1.0 - exp_w))

        # 3. Update SURFACE Elo
        if surf:
            exp_w_sf = expected_score(w_sf, l_sf)
            surface_elo[surf][w_id] = w_sf + k_w * (1.0 - exp_w_sf)
            surface_elo[surf][l_id] = l_sf + k_l * (0.0 - (1.0 - exp_w_sf))

        match_counts[w_id] += 1
        match_counts[l_id] += 1

        prev_w_hand = player_meta.get(w_id, (None, None, None))[2]
        prev_l_hand = player_meta.get(l_id, (None, None, None))[2]
        w_hand = getattr(row, "winner_hand", None)
        l_hand = getattr(row, "loser_hand", None)
        player_meta[w_id] = (row.tour, row.winner_name, w_hand if pd.notna(w_hand) else prev_w_hand)
        player_meta[l_id] = (row.tour, row.loser_name, l_hand if pd.notna(l_hand) else prev_l_hand)

    matches["w_pre_elo"] = pre_w_elo
    matches["l_pre_elo"] = pre_l_elo
    matches["w_pre_surf_elo"] = pre_w_surf
    matches["l_pre_surf_elo"] = pre_l_surf

    players_rows = []
    for pid, (tour, name, hand) in player_meta.items():
        players_rows.append({
            "player_id": pid,
            "tour": tour,
            "player_name": str(name),
            "hand": str(hand) if pd.notna(hand) else None,
            "matches_played": match_counts[pid],
            "elo_overall": round(overall_elo[pid], 2),
            "elo_hard": round(surface_elo["Hard"][pid], 2),
            "elo_clay": round(surface_elo["Clay"][pid], 2),
            "elo_grass": round(surface_elo["Grass"][pid], 2),
        })

    return matches, pd.DataFrame(players_rows)


def upload_players_to_supabase(players_df: pd.DataFrame, batch_size: int = 500):
    raw_records = players_df.to_dict(orient="records")
    records = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in raw_records
    ]
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        supabase.table("players").upsert(batch).execute()
    print(f"Successfully upserted {len(records)} players to Supabase.")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    matches_df = load_historical_matches()
    enriched_matches, players_df = compute_elo_ratings(matches_df)

    enriched_matches.to_csv("data/historical_matches_elo.csv", index=False)
    print("Saved local training dataset to data/historical_matches_elo.csv")

    upload_players_to_supabase(players_df)
