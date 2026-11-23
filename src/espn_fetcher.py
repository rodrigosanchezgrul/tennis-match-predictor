import json
import urllib.request
from datetime import datetime, timezone
import pandas as pd


def _infer_surface(tourney_name: str) -> str:
    name = (tourney_name or "").lower()
    clay_keywords = [
        "roland garros", "french open", "madrid", "rome", "internazionali",
        "monte-carlo", "monte carlo", "barcelona", "hamburg", "bastad",
        "gstaad", "kitzbuhel", "umag", "palermo", "rabat", "strasbourg",
        "charleston", "bogota", "rio", "buenos aires", "santiago", "estoril",
        "lyon", "geneva", "munich", "marrakech", "bucharest", "houston", "iasi"
    ]
    grass_keywords = [
        "wimbledon", "halle", "queen", "eastbourne", "mallorca",
        "hertogenbosch", "stuttgart", "bad homburg", "nottingham", "birmingham", "berlin"
    ]
    if any(k in name for k in clay_keywords):
        return "Clay"
    if any(k in name for k in grass_keywords):
        return "Grass"
    return "Hard"


def _infer_level(tourney_name: str) -> str:
    name = (tourney_name or "").lower()
    if any(g in name for g in ["australian open", "roland garros", "french open", "wimbledon", "us open"]):
        return "G"
    if any(m in name for m in ["masters", "1000", "indian wells", "miami", "madrid", "rome", "canada", "cincinnati", "shanghai", "paris", "doha", "dubai", "beijing", "wuhan"]):
        return "M"
    return "A"


def fetch_espn_matches(min_date_int: int = 20260601) -> pd.DataFrame:
    """Fetches completed ATP & WTA singles matches from ESPN's public scoreboard API."""
    now = datetime.now(timezone.utc)
    months = []
    y, m = 2026, 6
    while (y, m) <= (now.year, now.month):
        months.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1

    rows = []
    seen = set()

    for slug, tour_label in [("atp", "ATP"), ("wta", "WTA")]:
        for ym in months:
            url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/{slug}/scoreboard?dates={ym}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                print(f"[ESPN] Skipped {tour_label} {ym}: {exc}")
                continue

            for event in payload.get("events", []):
                t_name = event.get("name", "ATP/WTA Event")
                surface = _infer_surface(t_name)
                level = _infer_level(t_name)

                for grp in event.get("groupings", []):
                    draw_name = grp.get("grouping", {}).get("displayName", "")
                    if "Singles" not in draw_name:
                        continue

                    for comp in grp.get("competitions", []):
                        status_type = comp.get("status", {}).get("type", {})
                        if not status_type.get("completed", False):
                            continue

                        competitors = comp.get("competitors", [])
                        if len(competitors) != 2:
                            continue

                        winner_obj = next((c for c in competitors if c.get("winner") is True), None)
                        loser_obj = next((c for c in competitors if c.get("winner") is False), None)
                        if not winner_obj or not loser_obj:
                            continue

                        w_name = (winner_obj.get("athlete") or winner_obj.get("team") or {}).get("displayName")
                        l_name = (loser_obj.get("athlete") or loser_obj.get("team") or {}).get("displayName")
                        if not w_name or not l_name:
                            continue

                        raw_date = (comp.get("date") or "")[:10].replace("-", "")
                        if not raw_date.isdigit():
                            continue
                        date_int = int(raw_date)
                        if date_int <= min_date_int:
                            continue

                        key = (tour_label, date_int, w_name, l_name)
                        if key in seen:
                            continue
                        seen.add(key)

                        w_rank = (winner_obj.get("curatedRank") or {}).get("current")
                        l_rank = (loser_obj.get("curatedRank") or {}).get("current")
                        round_name = (comp.get("round") or {}).get("displayName") or "R32"

                        rows.append({
                            "tourney_id": f"ESPN-{ym}-{event.get('id', '')}",
                            "tourney_name": t_name,
                            "surface": surface,
                            "tourney_level": level,
                            "tourney_date": date_int,
                            "winner_name": w_name,
                            "loser_name": l_name,
                            "winner_rank": w_rank,
                            "loser_rank": l_rank,
                            "round": round_name,
                            "tour": tour_label,
                        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("tourney_date").reset_index(drop=True)
    print(f"[ESPN] Fetched {len(df)} new completed {min_date_int}+ ATP/WTA singles matches!")
    return df
