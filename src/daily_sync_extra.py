import json
import os
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def fetch_espn_rankings() -> dict:
    pts_map = {}
    for slug, tour in [("atp", "ATP"), ("wta", "WTA")]:
        url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/{slug}/rankings"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for block in data.get("rankings", []):
                for entry in block.get("ranks", []):
                    name = (entry.get("athlete") or {}).get("displayName") or ""
                    pts = int(float(entry.get("points") or 0))
                    if name and pts > 0:
                        pts_map[(tour, norm_name(name))] = pts
        except Exception as e:
            print(f"[WARN] ESPN rankings fetch failed ({tour}): {e}")
    return pts_map


def refresh_daily_player_stats():
    pts_map = fetch_espn_rankings()
    players = []
    start = 0
    while True:
        res = sb.table("players").select("player_id,tour,player_name,grand_slams,masters_1000,total_titles").range(start, start + 999).execute()
        if not res.data:
            break
        players.extend(res.data)
        if len(res.data) < 1000:
            break
        start += 1000

    cutoff_30 = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    cutoff_14 = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
    recent_res = sb.table("matches").select("tourney_date,tourney_name,round,winner_uid,loser_uid").gte("tourney_date", cutoff_30).order("tourney_date", desc=False).execute()
    recent_matches = recent_res.data or []

    cutoff_2 = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    hist = {}
    m14_count = {}
    new_finals = {}

    for m in recent_matches:
        dt = str(m["tourney_date"])[:10]
        w, l = m["winner_uid"], m["loser_uid"]
        hist.setdefault(w, []).append(1)
        hist.setdefault(l, []).append(0)
        if dt >= cutoff_14:
            m14_count[w] = m14_count.get(w, 0) + 1
            m14_count[l] = m14_count.get(l, 0) + 1
        if dt >= cutoff_2 and str(m.get("round") or "") == "F":
            tname = (m.get("tourney_name") or "").lower()
            is_g = any(g in tname for g in ["australian open", "roland garros", "french open", "wimbledon", "us open"])
            is_m = any(x in tname for x in ["masters", "1000", "indian wells", "miami", "madrid", "rome", "canada", "cincinnati", "shanghai", "paris", "doha", "dubai", "beijing", "wuhan"])
            rec = new_finals.setdefault(w, {"titles": 0, "slams": 0, "masters": 0})
            rec["titles"] += 1
            if is_g:
                rec["slams"] += 1
            elif is_m:
                rec["masters"] += 1

    updates = []
    for p in players:
        uid = p["player_id"]
        patch = {"player_id": uid}
        new_pts = pts_map.get((p["tour"], norm_name(p["player_name"])))
        if new_pts is not None:
            patch["ranking_points"] = new_pts
        if uid in hist:
            last10 = hist[uid][-10:]
            patch["form_10"] = round(sum(last10) / len(last10), 3)
        patch["matches_14d"] = m14_count.get(uid, 0)
        if uid in new_finals:
            patch["total_titles"] = int(p.get("total_titles") or 0) + new_finals[uid]["titles"]
            patch["grand_slams"] = int(p.get("grand_slams") or 0) + new_finals[uid]["slams"]
            patch["masters_1000"] = int(p.get("masters_1000") or 0) + new_finals[uid]["masters"]
        updates.append(patch)

    for i in range(0, len(updates), 500):
        sb.table("players").upsert(updates[i:i + 500], on_conflict="player_id").execute()
    print(f"Daily extra stats refreshed for {len(updates)} players.")


if __name__ == "__main__":
    refresh_daily_player_stats()
