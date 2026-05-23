"""
Playoff series-level prediction analysis.

Hypothesis: in playoffs, load management vanishes and rotations stabilize, so DUNCAN's
team-strength rating should align closer to true strength than in regular season.

Two analyses:
  1. Per-game accuracy on PLAYOFF games only (vs regular-season baseline).
  2. Per-SERIES accuracy: pre-series ratings -> predicted winner -> actual winner.

Playoff games are identified via season_flag==1 (end of regular season) snapshots in
duncan_ratings_with_standings.csv: any game after that date for a given season is playoff.
"""

from pathlib import Path
import numpy as np
import pandas as pd

NBA_DIR = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "dataset.csv"

HCA = 2.0


def load_rs_end_dates():
    standings = pd.read_csv(
        NBA_DIR / "duncan_ratings_with_standings.csv",
        parse_dates=["date"],
        usecols=["season", "date", "season_flag"],
    )
    rs_end = standings[standings["season_flag"] == 1].groupby("season")["date"].max()
    return rs_end.to_dict()


def label_playoffs(df, rs_end_map):
    df = df.copy()
    df["rs_end"] = df["season"].map(rs_end_map)
    df["is_playoff"] = df["game_date"] > df["rs_end"]
    df["is_playoff"] = df["is_playoff"].fillna(False)
    return df


def fmt_pct(x):
    return f"{x*100:.1f}%" if pd.notna(x) else "n/a"


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def per_game_split(df):
    section("Per-game accuracy: Regular season vs Playoffs")
    for label, sub in [("Regular season", df[~df["is_playoff"]]),
                       ("Playoffs",       df[df["is_playoff"]])]:
        n = len(sub)
        rmse_d = np.sqrt(np.mean(sub["duncan_err"] ** 2))
        rmse_v = np.sqrt(np.mean(sub["vegas_err"] ** 2))
        mae_d  = np.mean(np.abs(sub["duncan_err"]))
        mae_v  = np.mean(np.abs(sub["vegas_err"]))
        su = sub["duncan_su_correct"].mean()
        ats = sub[~sub["push"]]["duncan_ats_correct"].mean()
        gap_rmse = rmse_d - rmse_v
        print(f"{label}  n={n}")
        print(f"  RMSE:  DUNCAN {rmse_d:.2f}  vs  Vegas {rmse_v:.2f}   gap +{gap_rmse:.2f}")
        print(f"  MAE:   DUNCAN {mae_d:.2f}  vs  Vegas {mae_v:.2f}")
        print(f"  SU:    {fmt_pct(su)}")
        print(f"  ATS:   {fmt_pct(ats)}")


def detect_series(df):
    """Group playoff games into series by (season, unordered team pair).
       Real series have >=4 games; play-in games are single-elimination so they get filtered out."""
    po = df[df["is_playoff"]].copy()
    po["pair"] = po.apply(
        lambda r: tuple(sorted([r["home_name"], r["away_name"]])), axis=1
    )
    series_groups = po.groupby(["season", "pair"])
    rows = []
    for (season, pair), games in series_groups:
        n = len(games)
        if n < 2:
            continue  # play-in or odd single game
        games = games.sort_values("game_date")
        # Determine series winner: team with more wins in this set of games
        team_a, team_b = pair
        wins_a = 0
        wins_b = 0
        for _, g in games.iterrows():
            home_won = g["actual_home_margin"] > 0
            winner = g["home_name"] if home_won else g["away_name"]
            if winner == team_a:
                wins_a += 1
            else:
                wins_b += 1
        # First-to-4 = real playoff series
        if max(wins_a, wins_b) < 4:
            continue
        actual_winner = team_a if wins_a > wins_b else team_b
        actual_loser  = team_b if wins_a > wins_b else team_a
        series_score = f"{max(wins_a, wins_b)}-{min(wins_a, wins_b)}"

        # Pre-series ratings: rating snapshot before earliest game in the series
        first_game = games.iloc[0]
        rating_a = first_game["home_rating"] if first_game["home_name"] == team_a else first_game["away_rating"]
        rating_b = first_game["home_rating"] if first_game["home_name"] == team_b else first_game["away_rating"]
        duncan_fav = team_a if rating_a > rating_b else team_b
        duncan_correct = (duncan_fav == actual_winner)

        # Vegas: Game-1 spread tells us who Vegas favored. Negative home spread = home favored.
        g1_home = first_game["home_name"]
        g1_vegas_home_margin = first_game["vegas_home_margin"]
        vegas_fav = g1_home if g1_vegas_home_margin > 0 else (
            first_game["away_name"] if g1_vegas_home_margin < 0 else None
        )
        vegas_correct = (vegas_fav == actual_winner) if vegas_fav else None

        # Rating diff magnitude (how confident was DUNCAN?)
        rating_diff = abs(rating_a - rating_b)
        # Series round inference: by total playoff games-played order within the season
        rows.append({
            "season": int(season),
            "team_a": team_a, "team_b": team_b,
            "rating_a": rating_a, "rating_b": rating_b, "rating_diff": rating_diff,
            "duncan_fav": duncan_fav,
            "vegas_fav": vegas_fav,
            "vegas_g1_spread": g1_vegas_home_margin,
            "actual_winner": actual_winner,
            "actual_loser": actual_loser,
            "series_score": series_score,
            "n_games": n,
            "duncan_correct": duncan_correct,
            "vegas_correct": vegas_correct,
            "first_game_date": first_game["game_date"],
        })

    return pd.DataFrame(rows).sort_values(["season", "first_game_date"]).reset_index(drop=True)


def series_summary(series):
    section(f"Series-level: DUNCAN vs Vegas pick accuracy  (n={len(series)} series)")
    duncan_acc = series["duncan_correct"].mean()
    vegas_acc = series["vegas_correct"].dropna().mean()
    print(f"  DUNCAN picks series winner: {fmt_pct(duncan_acc)}")
    print(f"  Vegas (G1 fav) picks winner: {fmt_pct(vegas_acc)}  (n={series['vegas_correct'].notna().sum()})")
    # Where they agree vs disagree
    both = series[series["vegas_correct"].notna()].copy()
    both["agree"] = (both["duncan_fav"] == both["vegas_fav"])
    agree = both[both["agree"]]
    disagree = both[~both["agree"]]
    print(f"\n  When DUNCAN & Vegas agree (n={len(agree)}):  picks right {fmt_pct(agree['duncan_correct'].mean())}")
    if len(disagree):
        print(f"  When they disagree (n={len(disagree)}):")
        print(f"    DUNCAN's pick wins: {fmt_pct(disagree['duncan_correct'].mean())}")
        print(f"    Vegas's pick wins:  {fmt_pct(disagree['vegas_correct'].mean())}")


def series_by_confidence(series):
    section("Series accuracy by DUNCAN confidence (pre-series rating gap)")
    s = series.copy()
    bins = [0, 1.5, 3, 5, 7, 10, np.inf]
    s["gap_bucket"] = pd.cut(s["rating_diff"], bins=bins, right=False)
    by = s.groupby("gap_bucket", observed=True).agg(
        n=("duncan_correct", "size"),
        DUNCAN_acc=("duncan_correct", "mean"),
        Vegas_acc=("vegas_correct", "mean"),
    )
    print(by.to_string(formatters={
        "DUNCAN_acc": "{:.1%}".format,
        "Vegas_acc":  "{:.1%}".format,
    }))


def series_by_season(series):
    section("Series accuracy by season")
    by = series.groupby("season").agg(
        n=("duncan_correct", "size"),
        DUNCAN_acc=("duncan_correct", "mean"),
        Vegas_acc=("vegas_correct", "mean"),
    )
    print(by.to_string(formatters={
        "DUNCAN_acc": "{:.1%}".format,
        "Vegas_acc":  "{:.1%}".format,
    }))


def upsets_and_misses(series):
    section("DUNCAN's correct upset calls (DUNCAN picked underdog vs Vegas, won)")
    s = series.dropna(subset=["vegas_correct"]).copy()
    upsets = s[(s["duncan_fav"] != s["vegas_fav"]) & (s["duncan_correct"]) & (~s["vegas_correct"])]
    print(upsets[["season", "duncan_fav", "vegas_fav", "actual_winner", "series_score", "rating_diff", "vegas_g1_spread"]]
          .to_string(index=False, formatters={
              "rating_diff": "{:.2f}".format,
              "vegas_g1_spread": "{:+.1f}".format,
          }))
    section("DUNCAN's most overconfident misses (big rating gap, lost series)")
    misses = series[~series["duncan_correct"]].sort_values("rating_diff", ascending=False).head(15)
    print(misses[["season", "duncan_fav", "actual_winner", "series_score", "rating_diff", "vegas_g1_spread"]]
          .to_string(index=False, formatters={
              "rating_diff": "{:.2f}".format,
              "vegas_g1_spread": "{:+.1f}".format,
          }))


def main():
    df = pd.read_csv(DATA, parse_dates=["game_date"])
    df["season"] = df["season"].astype(int)
    rs_end_map = load_rs_end_dates()
    df = label_playoffs(df, rs_end_map)

    per_game_split(df)
    series = detect_series(df)
    series_summary(series)
    series_by_confidence(series)
    series_by_season(series)
    upsets_and_misses(series)

    out = Path(__file__).resolve().parent / "series.csv"
    series.to_csv(out, index=False)
    print(f"\nWrote series-level table -> {out}")


if __name__ == "__main__":
    main()
