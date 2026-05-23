"""
Analyze DUNCAN predictive accuracy: rating-only metrics + ATS performance vs Vegas open spread.

Run after build_dataset.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "dataset.csv"


def fmt_pct(x):
    return f"{x*100:.1f}%"


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def summary_metrics(df, label):
    n = len(df)
    if n == 0:
        print(f"{label}: empty")
        return
    rmse_d = np.sqrt(np.mean(df["duncan_err"] ** 2))
    mae_d  = np.mean(np.abs(df["duncan_err"]))
    rmse_v = np.sqrt(np.mean(df["vegas_err"] ** 2))
    mae_v  = np.mean(np.abs(df["vegas_err"]))
    su_acc = df["duncan_su_correct"].mean()
    home_win_rate = df["home_wins"].mean()
    ats_df = df[~df["push"]]
    ats_acc = ats_df["duncan_ats_correct"].mean() if len(ats_df) else float("nan")
    # Bias: do we systematically over- or under-predict home margin?
    duncan_bias = df["duncan_err"].mean()
    vegas_bias  = df["vegas_err"].mean()
    print(f"{label}  n={n}")
    print(f"  Margin RMSE: DUNCAN {rmse_d:.2f}  vs  Vegas {rmse_v:.2f}")
    print(f"  Margin MAE:  DUNCAN {mae_d:.2f}  vs  Vegas {mae_v:.2f}")
    print(f"  Mean err:    DUNCAN {duncan_bias:+.2f}  vs  Vegas {vegas_bias:+.2f}  (positive = over-predicts home margin)")
    print(f"  SU accuracy: DUNCAN {fmt_pct(su_acc)}   (home-win base rate {fmt_pct(home_win_rate)})")
    print(f"  ATS vs Vegas (pushes excl): {fmt_pct(ats_acc)}  on {len(ats_df)} bets")


def calibration_buckets(df):
    section("Calibration: predicted DUNCAN home margin -> actual mean home margin")
    bins = [-np.inf, -15, -10, -6, -3, 0, 3, 6, 10, 15, np.inf]
    df = df.copy()
    df["bucket"] = pd.cut(df["duncan_home_margin"], bins=bins)
    cal = df.groupby("bucket", observed=True).agg(
        n=("duncan_home_margin", "size"),
        predicted=("duncan_home_margin", "mean"),
        actual=("actual_home_margin", "mean"),
        home_win_rate=("home_wins", "mean"),
    )
    cal["resid"] = cal["actual"] - cal["predicted"]
    print(cal.to_string(formatters={
        "predicted": "{:+.2f}".format,
        "actual": "{:+.2f}".format,
        "resid": "{:+.2f}".format,
        "home_win_rate": "{:.1%}".format,
    }))


def ats_by_edge(df):
    section("ATS by |DUNCAN edge vs Vegas| (filter: only bet when DUNCAN disagrees with Vegas by N+ pts)")
    df = df[~df["push"]].copy()
    df["abs_edge"] = df["duncan_edge"].abs()
    bins = [0, 1, 2, 3, 4, 5, 6, 8, 10, np.inf]
    df["edge_bucket"] = pd.cut(df["abs_edge"], bins=bins, right=False)
    by = df.groupby("edge_bucket", observed=True).agg(
        n=("duncan_ats_correct", "size"),
        win_rate=("duncan_ats_correct", "mean"),
    )
    by["break_even_5275"] = by["win_rate"] - 0.5238  # break-even at -110 is 52.38%
    print(by.to_string(formatters={
        "win_rate": "{:.1%}".format,
        "break_even_5275": "{:+.1%}".format,
    }))


def by_season(df):
    section("Per-season metrics")
    rows = []
    for season, g in df.groupby("season"):
        n = len(g)
        rmse_d = np.sqrt(np.mean(g["duncan_err"] ** 2))
        rmse_v = np.sqrt(np.mean(g["vegas_err"] ** 2))
        su = g["duncan_su_correct"].mean()
        ats_g = g[~g["push"]]
        ats = ats_g["duncan_ats_correct"].mean()
        rows.append({
            "season": int(season), "n": n,
            "DUNCAN RMSE": rmse_d, "Vegas RMSE": rmse_v,
            "DUNCAN SU%": su, "DUNCAN ATS%": ats,
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, formatters={
        "DUNCAN RMSE": "{:.2f}".format,
        "Vegas RMSE": "{:.2f}".format,
        "DUNCAN SU%": "{:.1%}".format,
        "DUNCAN ATS%": "{:.1%}".format,
    }))


def directional_splits(df):
    section("ATS splits by direction")
    d = df[~df["push"]].copy()
    # DUNCAN-on-home (i.e. DUNCAN thinks home covers) vs DUNCAN-on-away
    home_picks = d[d["duncan_picks_home"]]
    away_picks = d[~d["duncan_picks_home"]]
    print(f"  DUNCAN picks home: n={len(home_picks)}  ATS%={home_picks['duncan_ats_correct'].mean():.1%}")
    print(f"  DUNCAN picks away: n={len(away_picks)}  ATS%={away_picks['duncan_ats_correct'].mean():.1%}")
    # Heavy favorites vs underdogs (by Vegas)
    fav = d[d["vegas_home_margin"].abs() >= 7]
    pk = d[d["vegas_home_margin"].abs() < 3]
    print(f"  Vegas spread >= 7  : n={len(fav)}  ATS%={fav['duncan_ats_correct'].mean():.1%}")
    print(f"  Vegas spread < 3   : n={len(pk)}   ATS%={pk['duncan_ats_correct'].mean():.1%}")


def best_worst_picks(df):
    section("Largest DUNCAN edges (biggest disagreements with Vegas)")
    d = df[~df["push"]].copy()
    d["abs_edge"] = d["duncan_edge"].abs()
    top = d.nlargest(15, "abs_edge")[[
        "game_date", "home_name", "away_name",
        "duncan_home_margin", "vegas_home_margin", "actual_home_margin",
        "duncan_edge", "duncan_ats_correct"
    ]]
    print(top.to_string(index=False, formatters={
        "duncan_home_margin": "{:+.2f}".format,
        "vegas_home_margin": "{:+.2f}".format,
        "actual_home_margin": "{:+.1f}".format,
        "duncan_edge": "{:+.2f}".format,
    }))


def main():
    df = pd.read_csv(DATA, parse_dates=["game_date"])
    df["season"] = df["season"].astype(int)
    section(f"DUNCAN predictive analysis  -  {len(df)} games, {df['season'].min()}-{df['season'].max()}")
    summary_metrics(df, "ALL")

    # Regular season vs playoffs (rough: playoffs = games after ~April 15)
    df["month"] = df["game_date"].dt.month
    reg = df[~((df["month"].isin([4, 5, 6])) & (df["game_date"].dt.day >= 15))]
    po  = df[((df["month"].isin([4, 5, 6])) & (df["game_date"].dt.day >= 15)) | (df["month"] == 6)]
    section("Regular season vs Playoffs (rough April-15+ cut)")
    summary_metrics(reg, "Regular season (approx)")
    summary_metrics(po,  "Playoffs (approx)")

    by_season(df)
    calibration_buckets(df)
    ats_by_edge(df)
    directional_splits(df)
    best_worst_picks(df)


if __name__ == "__main__":
    main()
