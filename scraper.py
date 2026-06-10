"""
Downloads game logs from the 30 NBA teams (regular season + playoffs)
using nba_api and produces two Excel files:

  1) raw_data_2026.xlsx       -> 1 stacked sheet with ALL teams,
                                 ready to train the predictive model
  2) post_loss_analysis_2026.xlsx -> summary of % wins by team

Training with the 30 teams teaches the model the GENERAL rule of what makes
a team win (not the idiosyncrasies of 2 teams), and that rule is then
applied to the NYK vs SAS prediction.

Requirements: pip install nba_api pandas openpyxl
"""

import time
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog

# --- Configuration ---
SEASON = "2025-26"      # nba_api format (regular season 2025-26)
# ALL 30 NBA teams (obtained from the nba_api catalog).
# To go back to only 2 teams: TEAM_ABBRS = ["NYK", "SAS"]
TEAM_ABBRS = sorted(t["abbreviation"] for t in teams.get_teams())
REQUEST_DELAY = 2.0     # 30 teams x 2 types = 60 calls. Be nice to stats.nba.com.
TIMEOUT = 60            # stats.nba.com is sometimes slow; generous timeout
MAX_RETRIES = 3         # retries if a call fails (timeout/network)

# --- SSL / proxy on the DGII network ---
# If your proxy breaks the certificate (CERTIFICATE_VERIFY_FAILED), leave False.
# This disables SSL verification for ALL nba_api calls.
VERIFY_SSL = False
# If the network requires an explicit proxy, put its address here (if not, leave None).
PROXY = None            # ex: "http://user:pass@proxy.dgii.gov.do:8080"

# --- Apply SSL/proxy config to nba_api for real ---
# nba_api uses requests internally with fixed verify=True. To force it to respect
# VERIFY_SSL/PROXY, you need to patch its HTTP layer and silence the warning.
import requests
import urllib3

if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Force verify (and proxy if applicable) in the requests session used by nba_api
_orig_request = requests.Session.request


def _request_with_ssl(self, *args, **kwargs):
    kwargs["verify"] = VERIFY_SSL
    if PROXY:
        kwargs.setdefault("proxies", {"http": PROXY, "https": PROXY})
    return _orig_request(self, *args, **kwargs)


requests.Session.request = _request_with_ssl
# Also for direct requests.get/post calls that don't use Session
_orig_get = requests.get


def _get_with_ssl(*args, **kwargs):
    kwargs["verify"] = VERIFY_SSL
    if PROXY:
        kwargs.setdefault("proxies", {"http": PROXY, "https": PROXY})
    return _orig_get(*args, **kwargs)


requests.get = _get_with_ssl

DATA_FILE = f"raw_data_{SEASON.split('-')[0]}.xlsx"
ANALYSIS_FILE = f"post_loss_analysis_{SEASON.split('-')[0]}.xlsx"


def get_team_id(abbr):
    """Resolves the NBA team_id from the abbreviation."""
    match = [t for t in teams.get_teams() if t["abbreviation"] == abbr]
    if not match:
        raise ValueError(f"Unknown abbreviation: {abbr}")
    return match[0]["id"]


def fetch_game_log(abbr):
    """Downloads game log from REGULAR SEASON + PLAYOFFS and combines them.
    Retries if a call fails. If the team didn't make playoffs,
    that part comes empty and is ignored without breaking."""
    team_id = get_team_id(abbr)
    frames = []
    for stype in ["Regular Season", "Playoffs"]:
        df = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                log = teamgamelog.TeamGameLog(
                    team_id=team_id, season=SEASON,
                    season_type_all_star=stype, timeout=TIMEOUT,
                )
                df = log.get_data_frames()[0]
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"    [!] {abbr} {stype}: failed after {MAX_RETRIES} "
                          f"attempts ({type(e).__name__}). Skipped.")
                else:
                    time.sleep(REQUEST_DELAY * attempt)  # increasing backoff
        time.sleep(REQUEST_DELAY)
        if df is None or df.empty:
            continue  # team without playoffs, or failure: skip
        df["is_playoff"] = 1 if stype == "Playoffs" else 0
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "TEAM", abbr)
    return out


def build_post_loss_detail(df):
    """
    Adds 'won' and 'prev_lost' columns to the game log.
    nba_api brings the 'WL' column with values 'W'/'L'.
    Note: nba_api returns games from most recent to oldest,
    so we reverse them so chronological order is correct.
    """
    df = df.copy()
    # Sort chronologically (ascending by date) so 'prev' makes sense
    if "GAME_DATE" in df.columns:
        # nba_api uses format 'MON DD, YYYY' (e.g.: 'OCT 31, 2025')
        df["_d"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y",
                                  errors="coerce")
        df = df.sort_values("_d").drop(columns="_d").reset_index(drop=True)
    df["won"] = df["WL"].astype(str).str.startswith("W")
    df["prev_lost"] = df["won"].shift(1).eq(False)
    return df


def analyze_post_loss(detail, abbr):
    """Calculates % wins overall vs. after loss."""
    overall = detail["won"].mean()
    post = detail[detail["prev_lost"]]
    post_wpct = post["won"].mean() if len(post) else float("nan")
    print(f"\n=== {abbr} ({SEASON}) ===")
    print(f"  Games: {len(detail)}")
    print(f"  % wins overall:      {overall:.1%}")
    print(f"  % wins after loss: {post_wpct:.1%} (n={len(post)})")
    print(f"  Bounce-back effect:            {post_wpct - overall:+.1%}")
    return {
        "Equipo": abbr,
        "Juegos": len(detail),
        "% Vict. general": round(overall, 3),
        "% Vict. tras derrota": round(post_wpct, 3),
        "N juegos tras derrota": len(post),
    }


def main():
    all_frames, detail_logs, summary = [], {}, []
    total = len(TEAM_ABBRS)
    ok, failed = 0, []

    print(f"Downloading {total} teams (regular season + playoffs)...")
    print(f"This takes ~2-3 min due to rate limit. Be patient.\n")

    for i, abbr in enumerate(TEAM_ABBRS, 1):
        print(f"[{i}/{total}] {abbr}...", end=" ", flush=True)
        try:
            df = fetch_game_log(abbr)
            if df is None or df.empty:
                print("no data, skipped")
                failed.append(abbr)
                continue
            all_frames.append(df)
            n_reg = (df["is_playoff"] == 0).sum()
            n_po = (df["is_playoff"] == 1).sum()
            print(f"{len(df)} games ({n_reg} reg + {n_po} playoff)")
            ok += 1
            # post-loss analysis by team (only for the analysis file)
            detail = build_post_loss_detail(df)
            detail_logs[abbr] = detail
            summary.append(analyze_post_loss_silent(detail, abbr))
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(abbr)

    if not all_frames:
        print("\n[ERROR] No team was downloaded. Check VERIFY_SSL/PROXY/network.")
        return None

    # Combine ALL teams in a single stacked DataFrame
    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal: {ok}/{total} teams, {len(combined)} games total.")
    if failed:
        print(f"Skipped: {', '.join(failed)}")

    # ---- FILE 1: RAW DATA (single stacked sheet for the model) ----
    with pd.ExcelWriter(DATA_FILE, engine="openpyxl") as w:
        combined.to_excel(w, sheet_name="all_teams", index=False)
    print(f"\n[OK] Raw data -> {DATA_FILE} "
          f"(1 stacked sheet: {len(combined)} rows, {combined['TEAM'].nunique()} teams)")

    # ---- FILE 2: POST-LOSS SUMMARY (all teams) ----
    if summary:
        with pd.ExcelWriter(ANALYSIS_FILE, engine="openpyxl") as w:
            pd.DataFrame(summary).sort_values("% Wins overall", ascending=False) \
                .to_excel(w, sheet_name="Summary", index=False)
        print(f"[OK] Analysis -> {ANALYSIS_FILE} (Summary of {ok} teams)")

    return combined


def analyze_post_loss_silent(detail, abbr):
    """Same as analyze_post_loss but without printing (it's 30 teams)."""
    overall = detail["won"].mean()
    post = detail[detail["prev_lost"]]
    post_wpct = post["won"].mean() if len(post) else float("nan")
    return {
        "Team": abbr,
        "Games": len(detail),
        "% Wins overall": round(overall, 3),
        "% Wins after loss": round(post_wpct, 3),
        "N games after loss": len(post),
    }
    


if __name__ == "__main__":
    main()
