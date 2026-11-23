import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo_engine import (
    compute_elo_ratings,
    load_historical_matches,
    upload_matches_to_supabase,
    upload_players_to_supabase,
)


def run_daily_sync():
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now_utc}] Starting daily ATP & WTA Elo sync...")

    matches_df = load_historical_matches()
    enriched_matches, players_df = compute_elo_ratings(matches_df)

    upload_players_to_supabase(players_df)
    upload_matches_to_supabase(enriched_matches)
    print(f"[{now_utc}] Daily sync complete!")


if __name__ == "__main__":
    run_daily_sync()
