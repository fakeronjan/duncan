"""
Title-odds model evaluation script.

Builds a logistic regression with:
  features: rating, rating_o, rating_d, season_progress,
            rating × progress, rating_o × progress, rating_d × progress

Trains via leave-one-season-out across seasons in [START, 2025]. Predicts
P(champion | features at this snapshot) for every alive team at every
snapshot. Eliminated teams hard-set to 0%; alive teams renormalized to
sum to 100% per snapshot.

Compares two training-era cutoffs (1984 and 2004) and reports:
  - Brier score (lower is better)
  - Log loss (lower is better)
  - Calibration (predicted bucket → actual rate)
  - Spot checks on famous seasons (Bulls 96, Pistons 04, Warriors 16)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime

GAMES_PER_RS = 82  # We treat short / lockout / bubble seasons as 82 too -
                   # season_progress saturates if a team played fewer games.

# Phase weights: alive team's CURRENT progress value
PHASE_RS_MAX        = 0.50
PHASE_POST_RS       = 0.55  # all 16 playoff teams; pre-R1
PHASE_R2_ENTRY      = 0.70  # won 1 series (in R2)
PHASE_CF_ENTRY      = 0.85  # won 2 series (in CF)
PHASE_FINALS_ENTRY  = 0.95  # won 3 series (in Finals)
PHASE_CHAMPION      = 1.00  # won 4 series (champion)


def _fit_logistic(X, y, reg=1e-3):
    """Plain LR via BFGS. Returns beta (intercept first). Mild L2 stabilizer
    (reg) keeps the optimization well-conditioned for the small number of
    'champion=1' cases."""
    n, k = X.shape
    Xa = np.column_stack([np.ones(n), X])

    def nll(beta):
        z = Xa @ beta
        return float(np.sum(np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - y * z) + reg * np.sum(beta[1:] ** 2))

    def grad(beta):
        z = Xa @ beta
        p_hat = 1.0 / (1.0 + np.exp(-z))
        g = Xa.T @ (p_hat - y)
        g[1:] += 2 * reg * beta[1:]
        return g

    res = minimize(nll, np.zeros(k + 1), jac=grad, method='BFGS',
                   options={'maxiter': 200, 'gtol': 1e-6})
    return res.x


def _predict_logistic(X, beta):
    Xa = np.column_stack([np.ones(X.shape[0]), X])
    z = Xa @ beta
    return 1.0 / (1.0 + np.exp(-z))


# ── 1. Load ratings + games ───────────────────────────────────────────────────
print("Loading data...")
ratings = pd.read_csv('duncan_ratings_with_standings.csv', parse_dates=['date'])
games   = pd.read_csv('all_nba_games.csv', parse_dates=['date_game'])
games   = games.sort_values('date_game').reset_index(drop=True)

# ── 2. Per-team RS-end date per season ───────────────────────────────────────
print("Computing per-season RS-end dates + champions...")
rs_end_by_season = {}
REGULAR_SEASON_GAMES = {1999: 50, 2012: 66, 2020: 72, 2021: 72}
for s, sg in games.groupby('season'):
    th = REGULAR_SEASON_GAMES.get(int(s), 82)
    home = sg[['date_game', 'home_team_name']].rename(columns={'home_team_name': 'team'})
    away = sg[['date_game', 'visitor_team_name']].rename(columns={'visitor_team_name': 'team'})
    all_games = pd.concat([home, away]).sort_values('date_game')
    all_games['n'] = all_games.groupby('team').cumcount() + 1
    thresh_dates = (all_games[all_games['n'] == th].groupby('team')['date_game'].first())
    if thresh_dates.empty:
        continue
    mode_date = thresh_dates.mode().iloc[0]
    delta = pd.Timedelta(days=2)
    in_window = thresh_dates[(thresh_dates >= mode_date - delta) & (thresh_dates <= mode_date + delta)]
    rs_end_by_season[int(s)] = in_window.max()

# ── 3. Bracket walk: per (season, team) list of series clinches ───────────────
# Re-uses the same per-series clinch threshold (4 H2H wins, BO7) - historical
# pre-1984 BO3 / 1984-2002 BO5 first-round series go unrecorded by this rule
# but the CHAMPION's path is BO7 in every era post-1976, so we still capture
# their series wins correctly. For non-champions, we may under-count R1
# series wins in older eras; in those cases the team's progress sits at 0.55
# longer than ideal. Accepted as a known limitation.
print("Walking brackets to identify series clinches...")
season_team_clinches = {}  # (season, team) -> list of clinch_date (sorted)
season_team_eliminated_date = {}  # (season, team) -> elimination date (None if never)
season_champ = {}
season_field = {}  # season -> set of teams in playoffs

for s, sg_all in games.groupby('season'):
    s = int(s)
    rs_end = rs_end_by_season.get(s)
    if rs_end is None:
        continue
    pg = sg_all[sg_all['date_game'] > rs_end].copy()
    if pg.empty:
        continue
    pg['_matchup'] = pg.apply(lambda r: tuple(sorted([r['home_team_name'], r['visitor_team_name']])), axis=1)
    history = {}
    for matchup, mg in pg.groupby('_matchup'):
        a, b = matchup
        mg_sorted = mg.sort_values('date_game').reset_index(drop=True)
        current_idx = [0]
        for i in range(1, len(mg_sorted)):
            gap = (mg_sorted.loc[i, 'date_game'] - mg_sorted.loc[i-1, 'date_game']).days
            if gap > 10:
                _proc(mg_sorted.iloc[current_idx], a, b, history) if False else None
                current_idx = [i]
            else:
                current_idx.append(i)
        # process the final sub-series
        def proc(sub, a, b, history):
            a_wins = (((sub['home_team_name']==a) & (sub['home_win']==1)) | ((sub['visitor_team_name']==a) & (sub['home_win']==0))).sum()
            b_wins = len(sub) - a_wins
            if a_wins >= 4 and a_wins > b_wins:
                winner, loser = a, b
            elif b_wins >= 4 and b_wins > a_wins:
                winner, loser = b, a
            else:
                return
            clinch = sub['date_game'].max()
            history.setdefault(winner, []).append((clinch, True, loser))
            history.setdefault(loser,  []).append((clinch, False, winner))
        proc(mg_sorted.iloc[current_idx], a, b, history)

    # Identify field = all teams that played a post-RS game
    season_field[s] = set(pg['home_team_name'].unique()) | set(pg['visitor_team_name'].unique())

    # For each team: clinch date list (sorted) + elimination date
    for team in season_field[s]:
        entries = sorted(history.get(team, []), key=lambda x: x[0])
        season_team_clinches[(s, team)] = [(d, w, o) for (d, w, o) in entries]
        # Eliminated when first 'loss' is recorded
        elim = next((d for (d, w, _) in entries if not w), None)
        season_team_eliminated_date[(s, team)] = elim

    # Champion = team with 4 series wins (alive, all wins)
    for team, entries in history.items():
        wins = [e for e in entries if e[1]]
        losses = [e for e in entries if not e[1]]
        if len(wins) >= 4 and not losses:
            season_champ[s] = team
            break

print(f"  Champions identified: {len(season_champ)} seasons")


# ── 4. games_played per (season, team, snapshot_date) ─────────────────────────
print("Computing games_played per (season, team, snapshot)...")
# Walk all games once per season, building cumulative count.
team_game_log = {}  # (season, team) -> sorted list of game dates
for s, sg in games.groupby('season'):
    s = int(s)
    for _, g in sg.iterrows():
        for t in [g['home_team_name'], g['visitor_team_name']]:
            team_game_log.setdefault((s, t), []).append(g['date_game'])
for k in team_game_log:
    team_game_log[k] = sorted(team_game_log[k])


def games_played(s, t, snap_date):
    """How many games has team t played in season s by the end of snap_date?"""
    log = team_game_log.get((s, t), [])
    # Binary search
    import bisect
    return bisect.bisect_right(log, snap_date)


# ── 5. Build training rows ────────────────────────────────────────────────────
print("Building training rows...")
ratings = ratings[ratings['rating'].notna() & ratings['rating_o'].notna() & ratings['rating_d'].notna()].copy()
ratings['date'] = pd.to_datetime(ratings['date'])

rows = []
for _, r in ratings.iterrows():
    s    = int(r['season'])
    team = r['name']
    sd   = r['date']
    rs_end = rs_end_by_season.get(s)
    if rs_end is None:
        continue

    # Determine alive + phase
    in_field = (s in season_field) and (team in season_field[s])
    if sd <= rs_end:
        # RS snapshot
        gp = games_played(s, team, sd)
        progress = PHASE_RS_MAX * min(gp / GAMES_PER_RS, 1.0)
        alive = True
    else:
        # Post-RS
        if not in_field:
            continue  # didn't make playoffs → eliminated, skip from training set
        elim = season_team_eliminated_date.get((s, team))
        # Series wins up to this snapshot
        clinches = season_team_clinches.get((s, team), [])
        series_won = sum(1 for (d, w, _) in clinches if d <= sd and w)
        # Determine alive status
        if elim is not None and sd >= elim:
            continue  # eliminated before this snapshot - skip
        # Map series_won to progress
        if series_won == 0:
            progress = PHASE_POST_RS
        elif series_won == 1:
            progress = PHASE_R2_ENTRY
        elif series_won == 2:
            progress = PHASE_CF_ENTRY
        elif series_won == 3:
            progress = PHASE_FINALS_ENTRY
        else:
            progress = PHASE_CHAMPION
        alive = True

    # Target
    is_champion = 1 if season_champ.get(s) == team else 0
    rows.append({
        'season':     s,
        'team':       team,
        'ranking_id': int(r['ranking_id']),
        'date':       sd,
        'rating':     float(r['rating']),
        'rating_o':   float(r['rating_o']),
        'rating_d':   float(r['rating_d']),
        'progress':   float(progress),
        'is_champion': is_champion,
    })

train_df = pd.DataFrame(rows)
print(f"  Training rows: {len(train_df):,}")
print(f"  Champion-positive rows: {int(train_df['is_champion'].sum()):,}")


def feature_matrix(d):
    """Features = [rating, rating_o, rating_d, progress,
                   rating*progress, rating_o*progress, rating_d*progress]."""
    p = d['progress'].values
    return np.column_stack([
        d['rating'].values,
        d['rating_o'].values,
        d['rating_d'].values,
        p,
        d['rating'].values * p,
        d['rating_o'].values * p,
        d['rating_d'].values * p,
    ])


# ── 6. LOO evaluation for both eras ───────────────────────────────────────────
def evaluate_era(start_year):
    """Train on seasons in [start_year, 2025]\\{s}, predict for s.
    Returns DataFrame with predictions + metrics."""
    print(f"\n=== Era: {start_year}+ ===")
    eligible = train_df[(train_df['season'] >= start_year) & (train_df['season'] <= 2025)].copy()
    seasons = sorted(eligible['season'].unique())
    print(f"  Seasons: {len(seasons)} ({seasons[0]}-{seasons[-1]})")

    predictions = []
    for s in seasons:
        train = eligible[eligible['season'] != s]
        held  = eligible[eligible['season'] == s]
        if held.empty or train.empty:
            continue
        X_train = feature_matrix(train)
        y_train = train['is_champion'].values.astype(float)
        beta = _fit_logistic(X_train, y_train)
        X_held = feature_matrix(held)
        p_raw = _predict_logistic(X_held, beta)
        # Normalize per ranking_id (alive teams sum to 1)
        held = held.copy()
        held['p_raw'] = p_raw
        held['p_norm'] = held.groupby('ranking_id')['p_raw'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0.0)
        predictions.append(held)
    preds = pd.concat(predictions, ignore_index=True)

    # Metrics
    y = preds['is_champion'].values
    p = preds['p_norm'].values
    p_clip = np.clip(p, 1e-9, 1 - 1e-9)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)))
    print(f"  Brier:    {brier:.6f}")
    print(f"  Log loss: {logloss:.5f}")
    return preds


def spot_check(preds, season, team, label):
    """Print P(champion) for `team` across the season at game milestones."""
    sub = preds[(preds['season'] == season) & (preds['team'] == team)].sort_values('date')
    if sub.empty:
        return
    print(f"\n  {label} - {team} {season}:")
    # Bin by progress level
    milestones = []
    for prog in [0.05, 0.20, 0.40, 0.50, 0.55, 0.70, 0.85, 0.95, 1.00]:
        row = sub[sub['progress'] >= prog - 0.02]
        if not row.empty:
            r = row.iloc[0]
            milestones.append((prog, r['p_norm'], r['date'].date()))
    seen = set()
    for prog, p, d in milestones:
        if prog in seen:
            continue
        seen.add(prog)
        print(f"    progress={prog:.2f}  P(champ)={p*100:.2f}%   ({d})")


preds_1984 = evaluate_era(1984)
preds_2004 = evaluate_era(2004)

# Calibration: 10 buckets
def calibration(preds, label):
    bins = np.linspace(0, 1, 11)
    p = preds['p_norm'].values
    y = preds['is_champion'].values
    print(f"\n  Calibration ({label}):")
    print(f"    {'p_bucket':>12}  {'count':>6}  {'pred_mean':>10}  {'actual':>8}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        print(f"    {f'[{lo:.2f},{hi:.2f})':>12}  {mask.sum():>6}  {p[mask].mean():>10.4f}  {y[mask].mean():>8.4f}")


calibration(preds_1984, "1984+")
calibration(preds_2004, "2004+")

# Spot checks
for preds, label in [(preds_1984, "1984+ model"), (preds_2004, "2004+ model")]:
    print(f"\n--- {label} spot checks ---")
    spot_check(preds, 1996, 'Chicago Bulls',         'Bulls (72-10, won)')
    spot_check(preds, 2004, 'Detroit Pistons',       'Pistons (defense champs)')
    spot_check(preds, 2016, 'Golden State Warriors', 'Warriors (73-9, did not win)')
    spot_check(preds, 2016, 'Cleveland Cavaliers',   'Cavaliers (won)')
    spot_check(preds, 2025, 'Oklahoma City Thunder', 'Thunder (won)')

# Per-snapshot sum verification: confirm alive teams sum to 100% per snapshot
print("\n--- Normalization verification (2016 NBA Finals, 2004+ model) ---")
finals_dates = ['2016-06-02','2016-06-05','2016-06-08','2016-06-10','2016-06-13','2016-06-16','2016-06-19']
for dt_str in finals_dates:
    dt = pd.Timestamp(dt_str)
    snap = preds_2004[(preds_2004['season'] == 2016) & (preds_2004['date'] == dt)]
    if snap.empty:
        continue
    total = snap['p_norm'].sum()
    print(f"  {dt_str}: {len(snap)} alive teams, P sum = {total*100:.2f}%")
    for _, r in snap.iterrows():
        print(f"    {r['team']:25s}  P(champ) = {r['p_norm']*100:.2f}%")

print("\nDone.")
