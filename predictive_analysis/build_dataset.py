"""
Build a per-game prediction dataset for 2020+ NBA seasons.

Joins:
  - DUNCAN game results (all_nba_games.csv)
  - DUNCAN pre-game ratings (duncan_ratings.csv)  -> snapshot with ranking_date < game_date
  - Historical Vegas open spreads (spreads_2020plus.csv, sourced from NBA_Betting v0.1.0-pre)

Output: predictive_analysis/dataset.csv with one row per game.
"""

import sys
from pathlib import Path
import pandas as pd

NBA_DIR = Path(__file__).resolve().parent.parent
SPREADS_CSV = Path("/tmp/nba_spreads/spreads_2020plus.csv")
OUT_PATH = Path(__file__).resolve().parent / "dataset.csv"

HCA = 2.0  # matches duncan.py HOME_COURT_ADJUSTMENT

CODE_TO_NAME = {
    "ATL": "Atlanta Hawks", "BKN": "Brooklyn Nets", "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


def main():
    games = pd.read_csv(NBA_DIR / "all_nba_games.csv", parse_dates=["date_game"])
    games = games[games["season"] >= 2020].copy()

    ratings = pd.read_csv(NBA_DIR / "duncan_ratings.csv", parse_dates=["ranking_date"])
    ratings = ratings[ratings["season"] >= 2020].copy()

    spreads = pd.read_csv(SPREADS_CSV, parse_dates=["game_date"])
    spreads["home_name"] = spreads["home_team"].map(CODE_TO_NAME)
    spreads["away_name"] = spreads["away_team"].map(CODE_TO_NAME)
    missing = spreads[spreads["home_name"].isna() | spreads["away_name"].isna()]
    if len(missing):
        print(f"WARN: {len(missing)} spread rows with unmapped team codes; dropping", file=sys.stderr)
    spreads = spreads.dropna(subset=["home_name", "away_name"]).copy()

    # For each (team, game_date), find rating from snapshot with max(ranking_date) < game_date.
    # Build a sorted list of (date, rating) per team, then merge_asof.
    # merge_asof requires both sides sorted by the `on` key globally.
    ratings_sorted = ratings.sort_values("ranking_date")

    def lookup_pregame(df, side):
        # df has game_date, plus a `<side>_name` column to join on
        side_col = f"{side}_name"
        df_s = df.sort_values("game_date").reset_index(drop=False)
        df_s = df_s[["index", "game_date", side_col]].rename(columns={side_col: "name"})
        merged = pd.merge_asof(
            df_s,
            ratings_sorted[["name", "ranking_date", "rating"]].rename(
                columns={"ranking_date": "game_date"}
            ),
            on="game_date",
            by="name",
            direction="backward",
            allow_exact_matches=False,  # strictly < game_date
        )
        return merged.set_index("index")["rating"].rename(f"{side}_rating")

    home_r = lookup_pregame(spreads, "home")
    away_r = lookup_pregame(spreads, "away")
    spreads = spreads.join(home_r).join(away_r)

    # Drop games where either pre-game rating is missing (early-season pre-window publish).
    pre = spreads.dropna(subset=["home_rating", "away_rating"]).copy()

    # Join actual margin from DUNCAN games to double-check scores + get unique_game_id.
    g = games[["date_game", "home_team_name", "visitor_team_name", "home_pts", "visitor_pts",
               "home_margin", "season", "unique_game_id"]].copy()
    g = g.rename(columns={
        "date_game": "game_date",
        "home_team_name": "home_name",
        "visitor_team_name": "away_name",
    })
    # ratings file's ranking_date stays as Timestamp throughout; ensure spread + game dates align
    g["game_date"] = pd.to_datetime(g["game_date"])
    pre["game_date"] = pd.to_datetime(pre["game_date"])
    merged = pre.merge(
        g, on=["game_date", "home_name", "away_name"], how="left",
        suffixes=("_sbr", "_duncan"),
    )

    # Sanity: spread DB and DUNCAN should agree on scores
    score_mismatch = merged[
        (merged["home_score"] != merged["home_pts"]) |
        (merged["away_score"] != merged["visitor_pts"])
    ]
    if len(score_mismatch):
        print(f"WARN: {len(score_mismatch)} games with score mismatch (sbr vs duncan)", file=sys.stderr)

    unmatched = merged["home_margin"].isna().sum()
    if unmatched:
        print(f"WARN: {unmatched} spread rows didn't match a DUNCAN game", file=sys.stderr)
    merged = merged.dropna(subset=["home_margin"]).copy()

    # Compute the metrics.
    # open_line: home perspective, negative = home favored. Predicted home margin by Vegas = -open_line.
    # DUNCAN predicted home margin = home_rating - away_rating + HCA.
    merged["vegas_home_margin"] = -merged["open_line"]
    merged["duncan_home_margin"] = merged["home_rating"] - merged["away_rating"] + HCA
    merged["actual_home_margin"] = merged["home_margin"]

    merged["duncan_err"] = merged["duncan_home_margin"] - merged["actual_home_margin"]
    merged["vegas_err"]  = merged["vegas_home_margin"]  - merged["actual_home_margin"]

    # Picks: who beats the spread by Vegas? DUNCAN picks home if its predicted margin > Vegas (i.e. it thinks home covers).
    # Cover: home covers if actual_home_margin > vegas_home_margin (home wins by more than spread).
    # Push: actual_home_margin == vegas_home_margin (rare with half-points)
    merged["home_covers"] = merged["actual_home_margin"] > merged["vegas_home_margin"]
    merged["push"] = merged["actual_home_margin"] == merged["vegas_home_margin"]
    merged["duncan_picks_home"] = merged["duncan_home_margin"] > merged["vegas_home_margin"]
    merged["duncan_edge"] = merged["duncan_home_margin"] - merged["vegas_home_margin"]
    merged["duncan_ats_correct"] = (
        ((merged["duncan_picks_home"]) & (merged["home_covers"])) |
        ((~merged["duncan_picks_home"]) & (~merged["home_covers"]))
    )
    merged.loc[merged["push"], "duncan_ats_correct"] = pd.NA  # ignore pushes

    # SU prediction.
    merged["duncan_picks_home_su"] = merged["duncan_home_margin"] > 0
    merged["home_wins"] = merged["actual_home_margin"] > 0
    merged["duncan_su_correct"] = merged["duncan_picks_home_su"] == merged["home_wins"]

    out_cols = [
        "unique_game_id", "season", "game_date",
        "home_name", "away_name",
        "home_pts", "visitor_pts", "actual_home_margin",
        "home_rating", "away_rating",
        "duncan_home_margin", "vegas_home_margin",
        "duncan_err", "vegas_err",
        "duncan_edge", "duncan_picks_home", "home_covers", "push",
        "duncan_ats_correct", "duncan_su_correct", "home_wins",
    ]
    out = merged[out_cols].sort_values(["game_date", "unique_game_id"]).reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} games -> {OUT_PATH}")
    print(f"  Seasons: {sorted(out['season'].unique())}")
    print(f"  Pushes:  {int(out['push'].sum())}")


if __name__ == "__main__":
    main()
