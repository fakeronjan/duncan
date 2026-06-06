"""
Title-odds v2 evaluation: extend the LR with two new features and compare
four variants in a clean LOO benchmark.

Variants:
  baseline   : rating, rating_o, rating_d, progress + 3 interactions
  +win%      : adds rs_win_pct
  +series    : adds series_wins, series_losses
  both       : all of the above

Reports Brier, log loss, calibration buckets, and spot checks at key
Finals snapshots (up 3-0, up 3-1, tied 2-2, down 0-3 etc.) to verify
the new features move predictions the way they should.
"""

import bisect
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── Phase weights (same as v1)
PHASE_RS_MAX        = 0.50
PHASE_POST_RS       = 0.55
PHASE_R2_ENTRY      = 0.70
PHASE_CF_ENTRY      = 0.85
PHASE_FINALS_ENTRY  = 0.95
PHASE_CHAMPION      = 1.00

GAMES_PER_RS = 82
TRAIN_FROM = 2004


def _fit_logistic(X, y, reg=1e-3):
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


def _predict(X, beta):
    Xa = np.column_stack([np.ones(X.shape[0]), X])
    z = Xa @ beta
    return 1.0 / (1.0 + np.exp(-z))


print("Loading data...")
ratings = pd.read_csv('duncan_ratings_with_standings.csv', parse_dates=['date'])
games   = pd.read_csv('all_nba_games.csv', parse_dates=['date_game'])
games   = games.sort_values('date_game').reset_index(drop=True)

# ── RS-end per season
REGULAR_SEASON_GAMES = {1999: 50, 2012: 66, 2020: 72, 2021: 72}
rs_end = {}
for s, sg in games.groupby('season'):
    th = REGULAR_SEASON_GAMES.get(int(s), 82)
    home = sg[['date_game', 'home_team_name']].rename(columns={'home_team_name': 'team'})
    away = sg[['date_game', 'visitor_team_name']].rename(columns={'visitor_team_name': 'team'})
    ag = pd.concat([home, away]).sort_values('date_game')
    ag['n'] = ag.groupby('team').cumcount() + 1
    thresh = ag[ag['n'] == th].groupby('team')['date_game'].first()
    if thresh.empty: continue
    mode = thresh.mode().iloc[0]
    d = pd.Timedelta(days=2)
    w = thresh[(thresh >= mode - d) & (thresh <= mode + d)]
    rs_end[int(s)] = w.max()

# ── Bracket walk: per-matchup series winner + game-by-game state
print("Walking brackets + computing per-snapshot series state...")
# Need per-(season, team, snapshot_date): in current series, series_wins, series_losses
# Strategy: for each season's playoff games, group into matchups (with date-split for
# 2020 round-robin etc.), order chronologically, then for each game emit cumulative
# (wins, losses) for both teams.
season_team_clinches = {}     # (s, team) -> [(clinch_date, won)]
season_team_eliminated = {}   # (s, team) -> elim date or None
season_field = {}             # s -> set of teams
season_champ = {}
# Per-team series-state events: (s, team) -> sorted list of (date, series_wins, series_losses, matchup_id)
# matchup_id distinguishes series across rounds (R1 series vs R2 series for same team)
series_state_events = {}      # (s, team) -> sorted list of (date, series_wins, series_losses)


def _proc_subseries(sub_df, a, b, history, series_state):
    aw = (((sub_df['home_team_name']==a)&(sub_df['home_win']==1))|((sub_df['visitor_team_name']==a)&(sub_df['home_win']==0))).sum()
    bw = len(sub_df) - aw
    # Walk games of this matchup in chrono order, emit (wins, losses) per team per game
    sub_sorted = sub_df.sort_values('date_game').reset_index(drop=True)
    a_w = b_w = 0
    for _, g in sub_sorted.iterrows():
        a_won = (g['home_team_name']==a and g['home_win']==1) or (g['visitor_team_name']==a and g['home_win']==0)
        if a_won: a_w += 1
        else:     b_w += 1
        series_state.setdefault(a, []).append((g['date_game'], a_w, b_w))
        series_state.setdefault(b, []).append((g['date_game'], b_w, a_w))
    # Determine series winner - era-aware threshold
    s_int = int(sub_sorted['season'].iloc[0])
    if   s_int >= 2003: clinch = 4
    elif s_int >= 1984: clinch = 3
    else:               clinch = 2
    if aw >= clinch and aw > bw:
        winner, loser = a, b
    elif bw >= clinch and bw > aw:
        winner, loser = b, a
    else:
        return
    last = sub_sorted['date_game'].max()
    history.setdefault(winner, []).append((last, True))
    history.setdefault(loser,  []).append((last, False))


for s, sg_all in games.groupby('season'):
    s = int(s)
    if s not in rs_end: continue
    pg = sg_all[sg_all['date_game'] > rs_end[s]].copy()
    if pg.empty: continue
    pg['_m'] = pg.apply(lambda r: tuple(sorted([r['home_team_name'], r['visitor_team_name']])), axis=1)
    history = {}
    series_state = {}  # team -> list of (date, series_wins, series_losses)
    real_field = set()
    for matchup, mg in pg.groupby('_m'):
        a, b = matchup
        if len(mg) < 3:
            continue  # play-in or stub
        mg_sorted = mg.sort_values('date_game').reset_index(drop=True)
        # Date-split into sub-series for 2020 bubble round-robin etc.
        cur = [0]
        for i in range(1, len(mg_sorted)):
            gap = (mg_sorted.loc[i, 'date_game'] - mg_sorted.loc[i-1, 'date_game']).days
            if gap > 10:
                _proc_subseries(mg_sorted.iloc[cur], a, b, history, series_state)
                cur = [i]
            else:
                cur.append(i)
        _proc_subseries(mg_sorted.iloc[cur], a, b, history, series_state)
        real_field.add(a); real_field.add(b)
    season_field[s] = real_field
    for team in real_field:
        entries = sorted(history.get(team, []), key=lambda x: x[0])
        season_team_clinches[(s, team)] = entries
        elim = next((d for (d, w) in entries if not w), None)
        season_team_eliminated[(s, team)] = elim
        # sort series state events per team
        ev = sorted(series_state.get(team, []), key=lambda x: x[0])
        series_state_events[(s, team)] = ev
    # Champion: all wins, no losses
    for team, entries in history.items():
        if entries and all(e[1] for e in entries):
            season_champ[s] = team
            break

# ── games_played + RS wins per (team, season, snapshot date)
print("Computing games_played + RS win pct per snapshot...")
# Walk all games once, build sorted-by-date lists per team
team_results = {}  # (s, team) -> list of (date, result)  where result = 1 for win, 0 for loss
for _, g in games.iterrows():
    s_int = int(g['season'])
    for t, is_home in [(g['home_team_name'], True), (g['visitor_team_name'], False)]:
        won = (g['home_win'] == 1) if is_home else (g['home_win'] == 0)
        team_results.setdefault((s_int, t), []).append((g['date_game'], int(won)))
for k in team_results:
    team_results[k] = sorted(team_results[k], key=lambda x: x[0])


def games_and_wins_played(s, t, snap_date):
    """Returns (games_played, wins) by team t in season s by end of snap_date,
    restricted to regular-season games (i.e. before season RS-end)."""
    lst = team_results.get((s, t), [])
    rs_end_dt = rs_end.get(s)
    if rs_end_dt is None:
        return 0, 0
    # Binary search by date
    dates = [x[0] for x in lst]
    idx = bisect.bisect_right(dates, snap_date)
    sub = lst[:idx]
    # Filter to RS-only
    sub_rs = [(d, w) for (d, w) in sub if d <= rs_end_dt]
    if not sub_rs:
        return 0, 0
    return len(sub_rs), sum(w for (_, w) in sub_rs)


def current_series_state(s, team, snap_date):
    """Returns (series_wins, series_losses) for team in their CURRENT active
    playoff series at snap_date. Looks at all series-state events ≤ snap_date
    and returns the last one whose matchup is still active (i.e. team hasn't
    been eliminated yet AND no later event for them exists for a different
    matchup). Simplification: just return the most recent event ≤ snap_date.
    If between rounds (no recent event), returns (0, 0)."""
    ev = series_state_events.get((s, team), [])
    if not ev:
        return 0, 0
    # Find latest event with date <= snap_date AND within last 30 days
    # (30-day cutoff handles "between rounds" - once a new round starts the
    # team gets fresh events).
    candidates = [(d, w, l) for (d, w, l) in ev if d <= snap_date]
    if not candidates:
        return 0, 0
    last_d, last_w, last_l = candidates[-1]
    # If this event is from a previous round (resolved series), team has
    # advanced and is between rounds → (0, 0). Detect by checking if there's
    # a clinch (entry in season_team_clinches) before snap_date whose date
    # is between event date and snap_date - but actually simpler: if the
    # last event is more than 14 days before snap_date, the series likely
    # ended and team is between rounds.
    days_since = (snap_date - last_d).days
    if days_since > 14:
        return 0, 0
    return last_w, last_l


# ── Build training rows
print("Building training rows...")
ratings = ratings[ratings['rating'].notna() & ratings['rating_o'].notna() & ratings['rating_d'].notna()].copy()
ratings['date'] = pd.to_datetime(ratings['date'])

rows = []
for _, r in ratings.iterrows():
    s_int = int(r['season'])
    team  = r['name']
    sd    = r['date']
    rs_end_dt = rs_end.get(s_int)
    if rs_end_dt is None:
        continue
    in_field = (s_int in season_field) and (team in season_field[s_int])
    if sd <= rs_end_dt:
        gp, w = games_and_wins_played(s_int, team, sd)
        progress = PHASE_RS_MAX * min(gp / GAMES_PER_RS, 1.0)
        rs_win_pct = (w / gp) if gp > 0 else 0.0
        series_w = 0
        series_l = 0
    else:
        if not in_field:
            continue
        elim = season_team_eliminated.get((s_int, team))
        if elim is not None and sd >= elim:
            continue
        clinches = season_team_clinches.get((s_int, team), [])
        series_won = sum(1 for (d, won) in clinches if d <= sd and won)
        if series_won == 0:   progress = PHASE_POST_RS
        elif series_won == 1: progress = PHASE_R2_ENTRY
        elif series_won == 2: progress = PHASE_CF_ENTRY
        elif series_won == 3: progress = PHASE_FINALS_ENTRY
        else:                 progress = PHASE_CHAMPION
        # RS win pct frozen at RS-end
        gp_rs, w_rs = games_and_wins_played(s_int, team, rs_end_dt)
        rs_win_pct = (w_rs / gp_rs) if gp_rs > 0 else 0.0
        series_w, series_l = current_series_state(s_int, team, sd)
    rows.append({
        'season': s_int, 'team': team, 'ranking_id': int(r['ranking_id']), 'date': sd,
        'rating': float(r['rating']), 'rating_o': float(r['rating_o']), 'rating_d': float(r['rating_d']),
        'progress': float(progress),
        'rs_win_pct': float(rs_win_pct),
        'series_w': int(series_w), 'series_l': int(series_l),
        'is_champion': 1 if season_champ.get(s_int) == team else 0,
    })

train_df = pd.DataFrame(rows)
print(f"  Rows: {len(train_df):,} ({int(train_df['is_champion'].sum())} champion-positive)")


# ── Feature builders for the 4 variants
def features_baseline(d):
    p = d['progress'].values
    return np.column_stack([
        d['rating'].values, d['rating_o'].values, d['rating_d'].values, p,
        d['rating'].values * p, d['rating_o'].values * p, d['rating_d'].values * p,
    ])

def features_win(d):
    base = features_baseline(d)
    return np.column_stack([base, d['rs_win_pct'].values])

def features_series(d):
    base = features_baseline(d)
    return np.column_stack([base, d['series_w'].values, d['series_l'].values])

def features_both(d):
    base = features_baseline(d)
    return np.column_stack([base, d['rs_win_pct'].values, d['series_w'].values, d['series_l'].values])


variants = [
    ('baseline', features_baseline),
    ('+win%',    features_win),
    ('+series',  features_series),
    ('both',     features_both),
]


def evaluate(name, feat_fn):
    eligible = train_df[train_df['season'] >= TRAIN_FROM].copy()
    seasons = sorted(s for s in eligible['season'].unique() if s in season_champ)
    preds = []
    for s in seasons:
        train = eligible[eligible['season'] != s]
        held  = eligible[eligible['season'] == s]
        if train.empty or held.empty:
            continue
        beta = _fit_logistic(feat_fn(train), train['is_champion'].values.astype(float))
        held = held.copy()
        held['p_raw'] = _predict(feat_fn(held), beta)
        held['p_norm'] = held.groupby('ranking_id')['p_raw'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0.0)
        preds.append(held)
    preds = pd.concat(preds, ignore_index=True)
    y = preds['is_champion'].values
    p = preds['p_norm'].values
    p_clip = np.clip(p, 1e-9, 1 - 1e-9)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)))
    return preds, brier, logloss


print()
print("="*60)
print("MODEL COMPARISON (2004+ LOO)")
print("="*60)
results = {}
for name, fn in variants:
    preds, brier, logloss = evaluate(name, fn)
    results[name] = preds
    print(f"  {name:10s}  Brier={brier:.5f}  LogLoss={logloss:.5f}")

print()
print("Calibration table (predicted bucket → actual rate)")
print()
print(f"  {'bucket':>12}  ", end='')
for name, _ in variants: print(f"{name:>12s}", end='')
print()
bins = np.linspace(0, 1, 11)
for lo, hi in zip(bins[:-1], bins[1:]):
    print(f"  [{lo:.2f},{hi:.2f})  ", end='')
    for name, _ in variants:
        p = results[name]['p_norm'].values
        y = results[name]['is_champion'].values
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            print(f"{'-':>12}", end='')
        else:
            actual = y[mask].mean()
            print(f"{actual:>12.3f}", end='')
    print()


print()
print("="*60)
print("SPOT CHECKS - 2016 Finals (Warriors vs Cavaliers)")
print("Cavs came back from 1-3 down to win 4-3.")
print("="*60)
fin_dates = ['2016-06-02','2016-06-05','2016-06-08','2016-06-10','2016-06-13','2016-06-16','2016-06-19']
for dt_str in fin_dates:
    dt = pd.Timestamp(dt_str)
    print(f"\n  {dt_str}:")
    for team in ['Cleveland Cavaliers', 'Golden State Warriors']:
        for name, _ in variants:
            snap = results[name][(results[name]['season']==2016) & (results[name]['date']==dt) & (results[name]['team']==team)]
            if snap.empty: continue
            p = snap['p_norm'].iloc[0]
            sw = snap['series_w'].iloc[0]
            sl = snap['series_l'].iloc[0]
            if name == 'baseline':
                print(f"    {team:25s}  {name:10s}  {p*100:>6.2f}%   (series {sw}-{sl})")
            else:
                print(f"    {' ':25s}  {name:10s}  {p*100:>6.2f}%")

print("\nDone.")
