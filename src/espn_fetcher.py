import calendar
import json
import re
import urllib.request
import zlib
from datetime import datetime, timezone
import pandas as pd


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _fallback_id(name: str) -> int:
    return 9000000 + (zlib.crc32(_norm_name(name).encode("utf-8")) % 999999)


def _infer_surface(tourney_name: str) -> str:
    name = (tourney_name or "").lower()
    clay = [
        "roland garros", "french open", "madrid", "rome", "internazionali",
        "monte-carlo", "monte carlo", "barcelona", "hamburg", "bastad",
        "gstaad", "kitzbuhel", "umag", "palermo", "rabat", "strasbourg",
        "charleston", "bogota", "rio", "buenos aires", "santiago", "estoril",
        "lyon", "geneva", "munich", "marrakech", "bucharest", "houston", "iasi"
    ]
    grass = [
        "wimbledon", "halle", "queen", "eastbourne", "mallorca",
        "hertogenbosch", "stuttgart", "bad homburg", "nottingham", "birmingham", "berlin"
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


def _format_score(winner_obj: dict, loser_obj: dict, status_detail: str) -> str:
    w_sets = winner_obj.get("linescores") or []
    l_sets = loser_obj.get("linescores") or []
    parts = []
    for sw, sl in zip(w_sets, l_sets):
        wg = sw.get("value")
        lg = sl.get("value")
        if wg is None or lg is None:
            continue
        set_str = f"{int(wg)}-{int(lg)}"
        tb_vals = [int(x) for x in (sw.get("tiebreak"), sl.get("tiebreak")) if x is not None]
        if tb_vals:
            set_str += f"({min(tb_vals)})"
        parts.append(set_str)
    if parts:
        if "ret" in (status_detail or "").lower():
            parts.append("RET")
        return " ".join(parts)
    return status_detail or ""


def fetch_espn_matches(min_date_int: int, name_to_id: dict) -> pd.DataFrame:
    """Fetches completed ATP & WTA singles matches from ESPN using YYYYMMDD-YYYYMMDD ranges."""
    now = datetime.now(timezone.utc)
    start_year = min_date_int // 10000
    start_month = (min_date_int // 100) % 100

    date_ranges = []
    y, m = start_year, start_month
    while (y, m) <= (now.year, now.month):
        last_day = calendar.monthrange(y, m)[1]
        start_str = f"{y}{m:02d}01"
        if (y, m) == (now.year, now.month):
            end_str = f"{now.year}{now.month:02d}{now.day:02d}"
        else:
            end_str = f"{y}{m:02d}{last_day:02d}"
        date_ranges.append(f"{start_str}-{end_str}")
        m += 1
        if m > 12:
            m = 1
            y += 1

    rows = []
    seen = set()
    opp_tour = {"ATP": "WTA", "WTA": "ATP"}

    for slug, tour_label in [("atp", "ATP"), ("wta", "WTA")]:
        for dr in date_ranges:
            url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/{slug}/scoreboard?dates={dr}&limit=100"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                print(f"Skipping ESPN {tour_label} {dr} ({exc})")
                continue

            for event in payload.get("events", []):
                t_name = event.get("name", f"{tour_label} Event")
                t_id = f"ESPN-{dr[:6]}-{event.get('id', '0')}"
                surface = _infer_surface(t_name)
                level = _infer_level(t_name)
                match_counter = 0

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
                    if not status_type.get("completed", False):
                        continue

                    competitors = comp.get("competitors", [])
                    if len(competitors) != 2:
                        continue

                    winner_obj = next((c for c in competitors if c.get("winner") is True), None)
                    loser_obj = next((c for c in competitors if c.get("winner") is False), None)
                    if not winner_obj or not loser_obj:
                        continue

                    w_name = (winner_obj.get("athlete") or winner_obj.get("team") or {}).get("displayName", "")
                    l_name = (loser_obj.get("athlete") or loser_obj.get("team") or {}).get("displayName", "")
                    if not w_name or not l_name or "TBD" in (w_name, l_name):
                        continue

                    nw, nl = _norm_name(w_name), _norm_name(l_name)
                    other = opp_tour[tour_label]
                    if ((tour_label, nw) not in name_to_id and (other, nw) in name_to_id) or (
                        (tour_label, nl) not in name_to_id and (other, nl) in name_to_id
                    ):
                        continue

                    raw_date = (comp.get("date") or "")[:10].replace("-", "")
                    if not raw_date.isdigit():
                        continue
                    date_int = int(raw_date)
                    if date_int <= min_date_int:
                        continue

                    dedup_key = (tour_label, date_int, nw, nl)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    match_counter += 1
                    w_id = name_to_id.get((tour_label, nw), _fallback_id(w_name))
                    l_id = name_to_id.get((tour_label, nl), _fallback_id(l_name))
                    score_str = _format_score(winner_obj, loser_obj, status_type.get("shortDetail", ""))

                    rows.append({
                        "tourney_id": t_id,
                        "tourney_name": t_name,
                        "surface": surface,
                        "tourney_level": level,
                        "tourney_date": date_int,
                        "match_num": match_counter,
                        "winner_id": w_id,
                        "winner_name": w_name,
                        "winner_hand": None,
                        "winner_rank": (winner_obj.get("curatedRank") or {}).get("current"),
                        "loser_id": l_id,
                        "loser_name": l_name,
                        "loser_hand": None,
                        "loser_rank": (loser_obj.get("curatedRank") or {}).get("current"),
                        "score": score_str,
                        "round": (comp.get("round") or {}).get("displayName") or "R32",
                        "tour": tour_label,
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"Loaded ESPN live matches (> {min_date_int}): {len(df)} matches")
    return df
