"""
Title-odds v3: replace raw (series_w, series_l) with a single feature -
log-odds of winning the current series under a coin-flip BO7 with era-
aware padding (BO5 → +1 to both sides, BO3 → +2 to both sides).

Compare:
  baseline      : v1 features
  +series_raw   : adds series_w, series_l (v2 finding)
  +series_logit : adds log-odds of BO7-conditional series-win probability
  +logit×prog   : adds series_logit AND series_logit × progress

Drops the +win% experiments (v2 found those hurt mid-Finals calibration).
"""

import bisect, math
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PHASE_RS_MAX       = 0.50
PHASE_POST_RS      = 0.55
PHASE_R2_ENTRY     = 0.70
PHASE_CF_ENTRY     = 0.85
PHASE_FINALS_ENTRY = 0.95
PHASE_CHAMPION     = 1.00
GAMES_PER_RS = 82
TRAIN_FROM = 2004


# ── BO7-conditional series-win probability (coin-flip remaining games)
# Built recursively: P(W, L) = 0.5 P(W+1, L) + 0.5 P(W, L+1)
def _build_bo7_table():
    """Recursive build of P(win series | up W wins, L losses) for BO7.
    Boundary: P(4, l) = 1 for l < 4; P(w, 4) = 0 for w < 4.
    Recursion: P(W, L) = 0.5 P(W+1, L) + 0.5 P(W, L+1)."""
    P = {}
    def compute(w, l):
        if (w, l) in P:
            return P[(w, l)]
        if w >= 4: P[(w, l)] = 1.0; return 1.0
        if l >= 4: P[(w, l)] = 0.0; return 0.0
        v = 0.5 * compute(w + 1, l) + 0.5 * compute(w, l + 1)
        P[(w, l)] = v
        return v
    for w in range(4):
        for l in range(4):
            compute(w, l)
    return P

BO7_P = _build_bo7_table()


def series_p_padded(w, l, clinch_threshold):
    """Probability of winning the series given state (w, l) and the era's
    clinch threshold (4 for BO7, 3 for BO5, 2 for BO3). Uses padding so a
    BO5 0-0 is treated as BO7 1-1, BO3 0-0 as BO7 2-2."""
    pad = 4 - clinch_threshold  # 0 for BO7, 1 for BO5, 2 for BO3
    w_p, l_p = min(w + pad, 4), min(l + pad, 4)
    if (w_p, l_p) in BO7_P:
        return BO7_P[(w_p, l_p)]
    # Out of grid (clinched) - clamp
    return 1.0 if w_p >= 4 else 0.0


def series_logit(w, l, clinch_threshold):
    p = series_p_padded(w, l, clinch_threshold)
    p_clip = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p_clip / (1 - p_clip))


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
    res = minimize(nll, np.zeros(k + 1), jac=grad, method='BFGS', options={'maxiter': 200, 'gtol': 1e-6})
    return res.x


def _predict(X, beta):
    Xa = np.column_stack([np.ones(X.shape[0]), X])
    z = Xa @ beta
    return 1.0 / (1.0 + np.exp(-z))


print("Loading data...")
ratings = pd.read_csv('duncan_ratings_with_standings.csv', parse_dates=['date'])
games   = pd.read_csv('all_nba_games.csv', parse_dates=['date_game'])
games   = games.sort_values('date_game').reset_index(drop=True)

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


print("Walking brackets + per-snapshot series state...")
season_team_clinches = {}
season_team_eliminated = {}
season_field = {}
season_champ = {}
series_state_events = {}  # (s, team) -> sorted list of (date, w, l)


def era_clinch(s):
    if s >= 2003: return 4
    if s >= 1984: return 3
    return 2


def _proc_subseries(sub_df, a, b, history, series_state):
    aw = (((sub_df['home_team_name']==a)&(sub_df['home_win']==1))|((sub_df['visitor_team_name']==a)&(sub_df['home_win']==0))).sum()
    bw = len(sub_df) - aw
    sub_sorted = sub_df.sort_values('date_game').reset_index(drop=True)
    a_w = b_w = 0
    for _, g in sub_sorted.iterrows():
        a_won = (g['home_team_name']==a and g['home_win']==1) or (g['visitor_team_name']==a and g['home_win']==0)
        if a_won: a_w += 1
        else:     b_w += 1
        series_state.setdefault(a, []).append((g['date_game'], a_w, b_w))
        series_state.setdefault(b, []).append((g['date_game'], b_w, a_w))
    s_int = int(sub_sorted['season'].iloc[0])
    clinch = era_clinch(s_int)
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
    series_state = {}
    real_field = set()
    for matchup, mg in pg.groupby('_m'):
        a, b = matchup
        if len(mg) < 3: continue
        mg_sorted = mg.sort_values('date_game').reset_index(drop=True)
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
        series_state_events[(s, team)] = sorted(series_state.get(team, []), key=lambda x: x[0])
    for team, entries in history.items():
        if entries and all(e[1] for e in entries):
            season_champ[s] = team
            break


def current_series_state(s, team, snap_date):
    ev = series_state_events.get((s, team), [])
    if not ev: return 0, 0
    cands = [(d, w, l) for (d, w, l) in ev if d <= snap_date]
    if not cands: return 0, 0
    last_d, last_w, last_l = cands[-1]
    if (snap_date - last_d).days > 14: return 0, 0
    return last_w, last_l


print("Building training rows...")
ratings = ratings[ratings['rating'].notna() & ratings['rating_o'].notna() & ratings['rating_d'].notna()].copy()
ratings['date'] = pd.to_datetime(ratings['date'])

rows = []
for _, r in ratings.iterrows():
    s_int = int(r['season'])
    team  = r['name']
    sd    = r['date']
    rs_end_dt = rs_end.get(s_int)
    if rs_end_dt is None: continue
    in_field = (s_int in season_field) and (team in season_field[s_int])
    if sd <= rs_end_dt:
        # Games played for progress
        lst = sorted([g for g in games[games['season']==s_int][['date_game','home_team_name','visitor_team_name']].itertuples(index=False) if (g[1]==team or g[2]==team) and g[0] <= sd], key=lambda x: x[0])
        gp = len(lst)
        progress = PHASE_RS_MAX * min(gp / GAMES_PER_RS, 1.0)
        series_w = series_l = 0
        series_lgt = 0.0  # neutral (log-odds of 0.5)
    else:
        if not in_field: continue
        elim = season_team_eliminated.get((s_int, team))
        if elim is not None and sd >= elim: continue
        clinches = season_team_clinches.get((s_int, team), [])
        series_won = sum(1 for (d, w) in clinches if d <= sd and w)
        if series_won == 0:   progress = PHASE_POST_RS
        elif series_won == 1: progress = PHASE_R2_ENTRY
        elif series_won == 2: progress = PHASE_CF_ENTRY
        elif series_won == 3: progress = PHASE_FINALS_ENTRY
        else:                 progress = PHASE_CHAMPION
        series_w, series_l = current_series_state(s_int, team, sd)
        # Era-aware: pre-2003 R1 was BO5 (clinch=3); pre-1984 R1 was BO3 (clinch=2)
        # series_won tells us the round; 0 means current series is R1.
        if series_won == 0:
            clinch = era_clinch(s_int)
        else:
            clinch = 4  # R2+ are BO7 in every era of DUNCAN data
        series_lgt = series_logit(series_w, series_l, clinch)
    rows.append({
        'season': s_int, 'team': team, 'ranking_id': int(r['ranking_id']), 'date': sd,
        'rating': float(r['rating']), 'rating_o': float(r['rating_o']), 'rating_d': float(r['rating_d']),
        'progress': float(progress),
        'series_w': int(series_w), 'series_l': int(series_l),
        'series_lgt': float(series_lgt),
        'is_champion': 1 if season_champ.get(s_int) == team else 0,
    })

train_df = pd.DataFrame(rows)
print(f"  Rows: {len(train_df):,} ({int(train_df['is_champion'].sum())} champion-positive)")


def features_baseline(d):
    p = d['progress'].values
    return np.column_stack([
        d['rating'].values, d['rating_o'].values, d['rating_d'].values, p,
        d['rating'].values * p, d['rating_o'].values * p, d['rating_d'].values * p,
    ])

def features_series_raw(d):
    return np.column_stack([features_baseline(d), d['series_w'].values, d['series_l'].values])

def features_series_logit(d):
    return np.column_stack([features_baseline(d), d['series_lgt'].values])

def features_logit_x_prog(d):
    base = features_baseline(d)
    return np.column_stack([base, d['series_lgt'].values, d['series_lgt'].values * d['progress'].values])


variants = [
    ('baseline',       features_baseline),
    ('+series_raw',    features_series_raw),
    ('+series_logit',  features_series_logit),
    ('+logit×prog',    features_logit_x_prog),
]


def evaluate(name, feat_fn):
    eligible = train_df[train_df['season'] >= TRAIN_FROM].copy()
    seasons = sorted(s for s in eligible['season'].unique() if s in season_champ)
    preds = []
    for s in seasons:
        train = eligible[eligible['season'] != s]
        held  = eligible[eligible['season'] == s]
        if train.empty or held.empty: continue
        beta = _fit_logistic(feat_fn(train), train['is_champion'].values.astype(float))
        held = held.copy()
        held['p_raw'] = _predict(feat_fn(held), beta)
        held['p_norm'] = held.groupby('ranking_id')['p_raw'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0.0)
        preds.append(held)
    preds = pd.concat(preds, ignore_index=True)
    y = preds['is_champion'].values
    p = preds['p_norm'].values
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    return preds, brier, logloss


print()
print("=" * 70)
print("MODEL COMPARISON (2004+ LOO)")
print("=" * 70)
results = {}
for name, fn in variants:
    preds, brier, logloss = evaluate(name, fn)
    results[name] = preds
    print(f"  {name:18s}  Brier={brier:.5f}  LogLoss={logloss:.5f}")

print()
print("Calibration table (predicted bucket → actual rate)")
print()
print(f"  {'bucket':>12}  ", end='')
for name, _ in variants: print(f"{name:>18s}", end='')
print()
bins = np.linspace(0, 1, 11)
for lo, hi in zip(bins[:-1], bins[1:]):
    print(f"  [{lo:.2f},{hi:.2f})  ", end='')
    for name, _ in variants:
        p = results[name]['p_norm'].values
        y = results[name]['is_champion'].values
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            print(f"{'-':>18}", end='')
        else:
            actual = y[mask].mean()
            n = mask.sum()
            print(f"{actual:.3f} (n={n:>4})".rjust(18), end='')
    print()


print()
print("=" * 70)
print("BO7 SERIES PROBABILITY TABLE (reference)")
print("=" * 70)
print(f"  state    P(win)   logit")
for total in range(7):
    for w in range(min(total, 3) + 1):
        l = total - w
        if l > 3 or w > 3: continue
        p = BO7_P[(w, l)]
        if 0 < p < 1:
            lg = math.log(p / (1 - p))
            print(f"  {w}-{l}      {p:.3f}    {lg:+.3f}")


print()
print("=" * 70)
print("SPOT CHECKS - 2016 Finals (Cavs came back from 1-3)")
print("=" * 70)
fin_dates = ['2016-06-02','2016-06-05','2016-06-08','2016-06-10','2016-06-13','2016-06-16','2016-06-19']
for dt_str in fin_dates:
    dt = pd.Timestamp(dt_str)
    print(f"\n  {dt_str}:")
    for team in ['Cleveland Cavaliers', 'Golden State Warriors']:
        snap = results['baseline'][(results['baseline']['season']==2016) & (results['baseline']['date']==dt) & (results['baseline']['team']==team)]
        if snap.empty: continue
        sw = snap['series_w'].iloc[0]
        sl = snap['series_l'].iloc[0]
        print(f"    {team:25s} (series {sw}-{sl})")
        for name, _ in variants:
            p = results[name][(results[name]['season']==2016) & (results[name]['date']==dt) & (results[name]['team']==team)]['p_norm'].iloc[0]
            print(f"      {name:18s}  {p*100:>6.2f}%")

print("\nDone.")
