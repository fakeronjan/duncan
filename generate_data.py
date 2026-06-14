"""
generate_data.py - reads duncan_ratings_with_standings.csv and writes JSON for the DUNCAN web frontend.
Run after duncan.py. Outputs to docs/data/.

Mirrors the LOBO/ZIDANE site architecture, with NBA-specific tweaks:
  - East/West conference mapping (per team, including historical relocations)
  - Single-year season display (e.g. "2025" = 2024-25 season per basketball-reference)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone
import re
from bisect import bisect_right

os.makedirs('docs/data/teams', exist_ok=True)
os.makedirs('docs/data/seasons', exist_ok=True)

print("Reading ratings...")
df = pd.read_csv('duncan_ratings_with_standings.csv')
df['date'] = pd.to_datetime(df['date']).dt.date

games = pd.read_csv('all_nba_games.csv')
games['date_game'] = pd.to_datetime(games['date_game']).dt.date


# ── NBA conference mapping (covers all team names since 1980) ────────────────
TEAM_CONFERENCE = {
    # Eastern Conference
    'Atlanta Hawks':        'East',
    'Boston Celtics':       'East',
    'Brooklyn Nets':        'East',
    'Buffalo Braves':       'East',
    'Charlotte Bobcats':    'East',
    'Charlotte Hornets':    'East',
    'Chicago Bulls':        'East',
    'Cleveland Cavaliers':  'East',
    'Detroit Pistons':      'East',
    'Indiana Pacers':       'East',
    'Miami Heat':           'East',
    'Milwaukee Bucks':      'East',
    'New Jersey Nets':      'East',
    'New York Knicks':      'East',
    'New York Nets':        'East',
    'Orlando Magic':        'East',
    'Philadelphia 76ers':   'East',
    'Toronto Raptors':      'East',
    'Washington Bullets':   'East',
    'Washington Wizards':   'East',

    # Western Conference
    'Dallas Mavericks':                  'West',
    'Denver Nuggets':                    'West',
    'Golden State Warriors':             'West',
    'Houston Rockets':                   'West',
    'Kansas City Kings':                 'West',
    'Los Angeles Clippers':              'West',
    'Los Angeles Lakers':                'West',
    'Memphis Grizzlies':                 'West',
    'Minnesota Timberwolves':            'West',
    'New Orleans Hornets':               'West',
    'New Orleans Jazz':                  'West',
    'New Orleans Pelicans':              'West',
    'New Orleans/Oklahoma City Hornets': 'West',
    'Oklahoma City Thunder':             'West',
    'Phoenix Suns':                      'West',
    'Portland Trail Blazers':            'West',
    'Sacramento Kings':                  'West',
    'San Antonio Spurs':                 'West',
    'San Diego Clippers':                'West',
    'Seattle SuperSonics':               'West',
    'Utah Jazz':                         'West',
    'Vancouver Grizzlies':               'West',
}


# Era-aware conference history. Teams listed here switched conferences during
# our coverage window (1977+). The 1980-81 realignment moved Bulls/Bucks/Pacers
# to the East and Spurs/Rockets to the West; the 1978-79 reshuffle moved
# Pistons East and Rockets-Pistons swap completed earlier. Lookup is by season
# (basketball-reference end-year convention).
TEAM_CONFERENCE_HISTORY = {
    'Chicago Bulls':       [(1977, 1980, 'West'), (1981, 9999, 'East')],
    'Milwaukee Bucks':     [(1977, 1980, 'West'), (1981, 9999, 'East')],
    'Indiana Pacers':      [(1977, 1979, 'West'), (1980, 9999, 'East')],
    'Detroit Pistons':     [(1977, 1978, 'West'), (1979, 9999, 'East')],
    'Houston Rockets':     [(1977, 1980, 'East'), (1981, 9999, 'West')],
    'San Antonio Spurs':   [(1977, 1980, 'East'), (1981, 9999, 'West')],
    'New Orleans Jazz':    [(1977, 1979, 'East')],  # franchise relocated to Utah after 1979
}


def conference(team, season=None):
    if season is not None:
        history = TEAM_CONFERENCE_HISTORY.get(team)
        if history:
            s = int(season)
            for start, end, conf in history:
                if start <= s <= end:
                    return conf
    return TEAM_CONFERENCE.get(team, 'Other')


# ── Era-aware display names ─────────────────────────────────────────────────
# duncan.py uses canonical (current) franchise names internally so a team's
# rating is continuous across same-market rebrands. Historical UI views
# (GOAT, Champions, Standings, per-team Season cells) should show what the
# team was actually called at the time. Maps canonical → list of
# (start_season, end_season_inclusive, display_name) ranges. 9999 = ongoing.
# Seasons follow basketball-reference convention: season N = N-1 to N (i.e.
# season 2014 = the 2013-14 NBA season).
NBA_TEAM_DISPLAY_HISTORY = {
    'Washington Wizards':    [(1977, 1997, 'Washington Bullets'),
                              (1998, 9999, 'Washington Wizards')],
    'Charlotte Hornets':     [(1989, 2002, 'Charlotte Hornets'),
                              (2005, 2014, 'Charlotte Bobcats'),
                              (2015, 9999, 'Charlotte Hornets')],
    'New Orleans Pelicans':  [(2003, 2005, 'New Orleans Hornets'),
                              (2006, 2007, 'NO/OKC Hornets'),
                              (2008, 2013, 'New Orleans Hornets'),
                              (2014, 9999, 'New Orleans Pelicans')],
}


def display_name(canonical, season):
    """Era-appropriate display name for the given canonical team and season."""
    history = NBA_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return canonical
    s = int(season)
    for start, end, name in history:
        if start <= s <= end:
            return name
    return canonical


def current_display_name(canonical):
    """The team's most recent display name (used for dropdowns / current snapshot)."""
    history = NBA_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return canonical
    return history[-1][2]


def historical_display_names(canonical):
    """Prior display names (most recent first), excluding the current name.
    Used to render '(formerly X / Y)' hints in the Team Summary dropdown."""
    history = NBA_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return []
    current = history[-1][2]
    seen = {current}
    out = []
    for _, _, name in reversed(history[:-1]):
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def clean(val):
    if pd.isna(val):
        return ''
    return str(val)


# duncan.py constructs last_match strings using the canonical franchise name
# (e.g. "W 117-116 vs. Charlotte Hornets" for a 2004-05 game when Charlotte
# was actually called the Bobcats). Rewrite the opponent portion with the
# era-appropriate display name so historical Team Summary / Standings views
# show the franchise's contemporary name. Format is "<W/L> <score> <vs.|@>
# <opponent>" - the opponent is the trailing portion (mirrors DILLON).
_LAST_MATCH_RE = re.compile(r'^([WLT])\s+(\d+\s*-\s*\d+)\s+(vs\.?(?:\s*\(N\))?|@)\s+(.+)$')

def era_aware_last_match(raw, season):
    if not raw:
        return raw
    m = _LAST_MATCH_RE.match(str(raw))
    if not m:
        return raw
    letter, score, venue, opponent = m.groups()
    return f"{letter} {score} {venue} {display_name(opponent.strip(), season)}"


def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')


def _od_fields(r):
    """Return rating_o/rating_d/rank_o/rank_d safely from a row. Returns
    None for missing values so downstream consumers (UI / JSON) can
    render '-' rather than '0'."""
    return {
        'rating_o': round(float(r['rating_o']), 3) if 'rating_o' in r and not pd.isna(r['rating_o']) else None,
        'rating_d': round(float(r['rating_d']), 3) if 'rating_d' in r and not pd.isna(r['rating_d']) else None,
        'rank_o':   int(r['rank_o']) if 'rank_o' in r and not pd.isna(r['rank_o']) else None,
        'rank_d':   int(r['rank_d']) if 'rank_d' in r and not pd.isna(r['rank_d']) else None,
    }


def _played(result):
    """True iff this row represents an actual game played. Upstream now
    writes empty strings for non-game-days (was 'No Game' previously) -
    both must be treated as "didn't play" or the forward-fill of last_match
    breaks for any snapshot date a team didn't play on."""
    if result is None or pd.isna(result):
        return False
    s = str(result).strip()
    return s not in ('', 'No Game')


# is_game_day: any row where the team actually played that snapshot date
df['is_game_day'] = df['last_game_result'].apply(_played).astype(int)
# is_end_of_season: collapse season_flag (1=last regular, 2=last postseason) to one boolean
df['is_end_of_season'] = df['season_flag'].isin([1, 2]).astype(int)

# Per-(team, season) forward-filled last game. Keying by season prevents
# cross-season carry-forward - at the start of a new season, teams that
# haven't played yet correctly show empty rather than their previous-season
# Finals result.
_last_game_history = {}
for (team, season), tdf in df[df['is_game_day'] == 1].sort_values('date').groupby(['name', 'season']):
    _last_game_history[(team, int(season))] = (
        [str(d) for d in tdf['date'].tolist()],
        tdf['last_game_result'].tolist(),
    )


def last_game_as_of(team, snap_date_str, season):
    entry = _last_game_history.get((team, int(season)))
    if not entry:
        return ''
    dates, games_list = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return games_list[idx] if idx >= 0 else ''


def last_game_date_as_of(team, snap_date_str, season):
    entry = _last_game_history.get((team, int(season)))
    if not entry:
        return ''
    dates, _ = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return dates[idx] if idx >= 0 else ''


# Per-season last regular-season date - used to flag playoff vs regular-season entries
_rs_end_dates = (
    df[df['season_flag'] == 1]
    .groupby('season')['date']
    .max()
    .to_dict()
)


def is_playoff(season, date_val):
    rs_end = _rs_end_dates.get(season)
    if rs_end is None:
        return False
    return date_val > rs_end


# Regular-season-end record per (team, season)
_reg_record_lookup = {
    (row['name'], int(row['season'])): row['record']
    for _, row in df[df['season_flag'] == 1].iterrows()
}

# End-of-playoffs combined record per (team, season). Used to derive the
# eventual playoff portion via _parse_record subtraction below - so GOAT
# rows show the team's playoff record regardless of which snapshot the row
# itself comes from (RS-end snapshots wouldn't otherwise know it).
_full_record_lookup = {
    (row['name'], int(row['season'])): row['record']
    for _, row in df[df['season_flag'] == 2].iterrows()
}


def _parse_record(rec):
    if not rec or pd.isna(rec):
        return None
    m = re.match(r'(\d+)\s*-\s*(\d+)', str(rec))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def playoff_record(full_record, regular_record):
    f = _parse_record(full_record)
    r = _parse_record(regular_record)
    if not f or not r:
        return ''
    pw, pl = f[0] - r[0], f[1] - r[1]
    if pw < 0 or pl < 0:
        return ''
    return f"{pw}-{pl}"


# ── Title Odds (logistic regression, leave-one-season-out, 2004+) ────────────
# Continuous-progress model: P(champion | rating, rating_o, rating_d,
# season_progress, plus 3 rating×progress interactions). Phase-weighted
# progress: linear 0→0.5 during RS (games_played/82), then jumps at each
# playoff round (post-RS = 0.55, post-R1 = 0.70, post-R2 = 0.85,
# post-CF = 0.95, crowned = 1.0). Eliminated teams hard-set to 0%; alive
# teams renormalized to sum to 100% per snapshot. Training cutoff at
# 2004+ (24 modern seasons) gave the best-calibrated predictions in
# evaluation - pre-2004 dynasty data made the model overconfident on
# mid-range probabilities.
print("Computing title odds (logistic regression, leave-one-season-out)...")
from scipy.optimize import minimize

GAMES_PER_RS_TO = 82
PHASE_RS_MAX_TO        = 0.50
PHASE_POST_RS_TO       = 0.55
PHASE_R2_ENTRY_TO      = 0.70
PHASE_CF_ENTRY_TO      = 0.85
PHASE_FINALS_ENTRY_TO  = 0.95
PHASE_CHAMPION_TO      = 1.00
TITLE_TRAIN_FROM_SEASON = 2004  # no upper bound - every newly-completed
                                # season auto-joins the training pool on the
                                # next cron run, mirroring DILLON's pattern.

# Seasons with NBA Play-In Tournament. 2020 was the bubble's 1-game
# play-in for the 8 seed; 2021+ is the formal 4-team tournament (7v8 and
# 9v10, then loser-of-7v8 vs winner-of-9v10 for the 8 seed). All play-in
# matchups are BO1 - a single game. Used by the bracket walker below to
# distinguish a real series (BO5 / BO7) from a play-in matchup. Add the
# new season here when it becomes an in-progress season.
PLAY_IN_SEASONS = {2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027}

# RS-end dates per season (mode-of-threshold game per team, +/- 2 days)
REGULAR_SEASON_GAMES_TO = {1999: 50, 2012: 66, 2020: 72, 2021: 72}
games_to = games.copy()
games_to['date_game'] = pd.to_datetime(games_to['date_game'])
_to_rs_end = {}
for s, sg in games_to.groupby('season'):
    th = REGULAR_SEASON_GAMES_TO.get(int(s), 82)
    home = sg[['date_game', 'home_team_name']].rename(columns={'home_team_name': 'team'})
    away = sg[['date_game', 'visitor_team_name']].rename(columns={'visitor_team_name': 'team'})
    ag = pd.concat([home, away]).sort_values('date_game')
    ag['n'] = ag.groupby('team').cumcount() + 1
    thresh = ag[ag['n'] == th].groupby('team')['date_game'].first()
    if thresh.empty:
        continue
    mode = thresh.mode().iloc[0]
    d = pd.Timedelta(days=2)
    w = thresh[(thresh >= mode - d) & (thresh <= mode + d)]
    _to_rs_end[int(s)] = w.max()

# Bracket walk: per-season clinch dates + elimination dates + per-game series state
_to_clinches = {}             # (season, team) -> sorted list of (date, won)
_to_eliminated = {}           # (season, team) -> elimination date (None if never)
_to_field = {}                # season -> set of teams in playoffs
_to_series_events = {}        # (season, team) -> sorted list of (date, series_wins, series_losses)


def _to_clinch_threshold(season):
    """Era-aware first-round clinch threshold (NBA R1 format moved from BO3
    to BO5 to BO7 over the data window). For 2003+ all rounds are BO7 so
    the BO7 4-win threshold also correctly handles in-progress series at
    1-0 / 2-1 (won't falsely mark as decided). For 1984-2002 the threshold
    of 3 captures both BO5 R1 series and BO7 later rounds in closed
    historical data (no in-progress R2+ snapshots in those years). For
    pre-1984 the BO3 R1 threshold is 2."""
    s = int(season)
    if s >= 2003:
        return 4
    if s >= 1984:
        return 3
    return 2


def _to_proc_series(sub, a, b, history, season, series_state=None):
    """Process a single playoff series (a sub-frame of games between team a
    and team b). Records the series-clinch outcome in `history` (used for
    elimination + championship detection) AND, if `series_state` is given,
    emits per-game (date, wins, losses) events for both teams (used downstream
    to score the current-series feature for the title-odds model).
    """
    aw = (((sub['home_team_name']==a)&(sub['home_win']==1))|((sub['visitor_team_name']==a)&(sub['home_win']==0))).sum()
    bw = len(sub) - aw
    if series_state is not None:
        sub_sorted = sub.sort_values('date_game').reset_index(drop=True)
        a_w = b_w = 0
        for _, g in sub_sorted.iterrows():
            a_won = (g['home_team_name']==a and g['home_win']==1) or (g['visitor_team_name']==a and g['home_win']==0)
            if a_won: a_w += 1
            else:     b_w += 1
            series_state.setdefault(a, []).append((g['date_game'], a_w, b_w))
            series_state.setdefault(b, []).append((g['date_game'], b_w, a_w))
    clinch = _to_clinch_threshold(season)
    if aw >= clinch and aw > bw:
        winner, loser = a, b
    elif bw >= clinch and bw > aw:
        winner, loser = b, a
    else:
        return
    cd = sub['date_game'].max()
    history.setdefault(winner, []).append((cd, True))
    history.setdefault(loser,  []).append((cd, False))


for s, sg_all in games_to.groupby('season'):
    s = int(s)
    rs_end_dt = _to_rs_end.get(s)
    if rs_end_dt is None:
        continue
    pg = sg_all[sg_all['date_game'] > rs_end_dt].copy()
    if pg.empty:
        continue
    pg['_m'] = pg.apply(lambda r: tuple(sorted([r['home_team_name'], r['visitor_team_name']])), axis=1)
    history = {}
    series_state = {}   # team -> [(date, series_wins, series_losses), ...]
    # "Real bracket" = teams in actual playoff series (BO5 / BO7), as
    # opposed to play-in tournament matchups (BO1). The old filter used
    # `len(mg) < 3` as a proxy for "play-in," but that also swept up
    # in-progress real series at games 1-2 (cost us the 2026 Knicks-Spurs
    # 2-0 Finals read). Switched to an explicit play-in-season check:
    # skip exactly the 1-game matchups in seasons that have a play-in
    # tournament, leave 2-game in-progress series alone.
    real_field = set()
    last_post_rs_date_for_team = {}  # used as elim fallback for play-in losers
    for matchup, mg in pg.groupby('_m'):
        a, b = matchup
        if s in PLAY_IN_SEASONS and len(mg) == 1:
            continue  # play-in matchup (BO1), not a real series
        mg_s = mg.sort_values('date_game').reset_index(drop=True)
        cur = [0]
        for i in range(1, len(mg_s)):
            gap = (mg_s.loc[i, 'date_game'] - mg_s.loc[i-1, 'date_game']).days
            if gap > 10:
                _to_proc_series(mg_s.iloc[cur], a, b, history, s, series_state)
                cur = [i]
            else:
                cur.append(i)
        _to_proc_series(mg_s.iloc[cur], a, b, history, s, series_state)
        real_field.add(a)
        real_field.add(b)
    _to_field[s] = real_field
    for team in real_field:
        entries = sorted(history.get(team, []), key=lambda x: x[0])
        _to_clinches[(s, team)] = entries
        elim = next((d for (d, w) in entries if not w), None)
        _to_eliminated[(s, team)] = elim
        _to_series_events[(s, team)] = sorted(series_state.get(team, []), key=lambda x: x[0])

# Champion per season: the team whose bracket history is all wins, no
# losses (they advanced through every series they played). Works across
# eras even though the number of series varies - 4 in modern bracket, 3
# for bye'd top seeds in the pre-1984 12-team format.
_to_champion = {}
for (s, team), entries in _to_clinches.items():
    if entries and all(e[1] for e in entries):
        _to_champion[s] = team

# games_played(season, team, snap_date)
_to_game_log = {}  # (season, team) -> sorted list of game dates
for _, g in games_to.iterrows():
    s_int = int(g['season'])
    for t in (g['home_team_name'], g['visitor_team_name']):
        _to_game_log.setdefault((s_int, t), []).append(g['date_game'])
for k in _to_game_log:
    _to_game_log[k] = sorted(_to_game_log[k])


def _to_games_played(s, t, snap_date):
    log = _to_game_log.get((s, t), [])
    return bisect_right(log, snap_date)


def _to_current_series_state(s, team, snap_date):
    """Return (series_wins, series_losses) for `team`'s CURRENT active
    playoff series at snap_date. Returns (0, 0) for RS / between-rounds /
    no-recent-event snapshots. A 14-day gap from the last recorded event
    means the team has advanced and isn't in an active series yet. On a
    clinch day itself, the team has also advanced - returns (0, 0) so the
    snapshot reads as "post-round" rather than "still in round at series_w
    = clinch threshold". Mirrors the GRIFFEY clinch-day fix; without this
    the LR features look like (advanced progress, max series_w) which is a
    rare pattern the model can't score consistently and produces spurious
    mid-bracket flips (2016 GSW vs CLE on CF clinch day surfaced this)."""
    ev = _to_series_events.get((s, team), [])
    if not ev:
        return 0, 0
    cands = [(d, w, l) for (d, w, l) in ev if d <= snap_date]
    if not cands:
        return 0, 0
    last_d, last_w, last_l = cands[-1]
    if (snap_date - last_d).days > 14:
        return 0, 0
    # If the last event corresponds to a clinch entry for this team, the
    # series ended on last_d and the team has advanced. Check against
    # _to_clinches (series-aware - doesn't rely on era-wide threshold).
    clinches = _to_clinches.get((s, team), [])
    if any(d == last_d for (d, _) in clinches):
        return 0, 0
    return last_w, last_l


def _to_series_padded(w, l, season, series_won):
    """Apply era-aware padding so series state is reported in BO7-equivalent
    space - BO5 series get +1 wins and +1 losses, BO3 series get +2 each.
    Only the first round (series_won == 0) needs padding in historical eras;
    all other rounds were BO7. This lets the 2004+-trained model "see" a
    historical R1 BO5 series 3-1 as the equivalent BO7 state 4-2 (sweep
    end) and apply the right coefficient."""
    if series_won == 0:
        clinch = _to_clinch_threshold(season)
    else:
        clinch = 4
    pad = 4 - clinch  # 0 for BO7, 1 for BO5, 2 for BO3
    return w + pad, l + pad


# Build training/prediction rows
_to_df = df[df['rating_o'].notna() & df['rating_d'].notna()].copy()
_to_df['date_dt'] = pd.to_datetime(_to_df['date'])

_to_rows = []
for _, r in _to_df.iterrows():
    s_int = int(r['season'])
    team  = r['name']
    sd    = r['date_dt']
    rs_end_dt = _to_rs_end.get(s_int)
    if rs_end_dt is None:
        continue
    in_field = (s_int in _to_field) and (team in _to_field[s_int])
    series_w = series_l = 0
    # Strict `<` so the rs_end snapshot itself goes through the in_field
    # gate below - non-playoff teams correctly drop out (cache will return
    # null, UI renders '-') rather than carrying tiny LR-induced
    # probabilities that round to 0.0%. PS teams pick up PHASE_POST_RS_TO
    # (0.55) via series_won==0 at the bottom of the else branch.
    if sd < rs_end_dt:
        gp = _to_games_played(s_int, team, sd)
        progress = PHASE_RS_MAX_TO * min(gp / GAMES_PER_RS_TO, 1.0)
    else:
        if not in_field:
            continue
        elim = _to_eliminated.get((s_int, team))
        if elim is not None and sd >= elim:
            continue
        clinches = _to_clinches.get((s_int, team), [])
        series_won = sum(1 for (d, w) in clinches if d <= sd and w)
        if series_won == 0:
            progress = PHASE_POST_RS_TO
        elif series_won == 1:
            progress = PHASE_R2_ENTRY_TO
        elif series_won == 2:
            progress = PHASE_CF_ENTRY_TO
        elif series_won == 3:
            progress = PHASE_FINALS_ENTRY_TO
        else:
            progress = PHASE_CHAMPION_TO
        raw_w, raw_l = _to_current_series_state(s_int, team, sd)
        series_w, series_l = _to_series_padded(raw_w, raw_l, s_int, series_won)
    _to_rows.append({
        'season': s_int, 'team': team, 'ranking_id': int(r['ranking_id']),
        'rating': float(r['rating']), 'rating_o': float(r['rating_o']),
        'rating_d': float(r['rating_d']), 'progress': float(progress),
        'series_w': int(series_w), 'series_l': int(series_l),
        'is_champion': 1 if _to_champion.get(s_int) == team else 0,
    })

_to_train_df = pd.DataFrame(_to_rows)
print(f"  Title-odds training rows: {len(_to_train_df):,} "
      f"({int(_to_train_df['is_champion'].sum())} champion-positive)")


def _to_features(d):
    """Title-odds feature matrix. Includes the per-snapshot current-series
    state (series_w, series_l) - values are era-padded so a BO5 3-1 reads
    as BO7 4-2. For non-playoff / between-rounds snapshots, both are 0."""
    p = d['progress'].values
    return np.column_stack([
        d['rating'].values, d['rating_o'].values, d['rating_d'].values,
        p,
        d['rating'].values * p,
        d['rating_o'].values * p,
        d['rating_d'].values * p,
        d['series_w'].values,
        d['series_l'].values,
    ])


def _to_fit_logistic(X, y, reg=1e-3):
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


def _to_predict_logistic(X, beta):
    Xa = np.column_stack([np.ones(X.shape[0]), X])
    z = Xa @ beta
    return 1.0 / (1.0 + np.exp(-z))


# LOO across modern seasons (2004+ with known champion). For in-progress
# current seasons, predict with full-history-trained model.
_eligible = _to_train_df[
    _to_train_df['season'] >= TITLE_TRAIN_FROM_SEASON
].copy()
_completed_seasons = {s for s in _eligible['season'].unique() if s in _to_champion}

_title_odds_cache = {}  # (ranking_id, team) -> float
for s_int in _completed_seasons:
    train = _eligible[_eligible['season'] != s_int]
    held  = _eligible[_eligible['season'] == s_int]
    if train.empty or held.empty:
        continue
    beta = _to_fit_logistic(_to_features(train), train['is_champion'].values.astype(float))
    held = held.copy()
    held['p_raw'] = _to_predict_logistic(_to_features(held), beta)
    held['p_norm'] = held.groupby('ranking_id')['p_raw'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0.0)
    for _, r in held.iterrows():
        _title_odds_cache[(r['ranking_id'], r['team'])] = float(r['p_norm'])

# In-progress / current-season predictions: train on the full 2004+ eligible
# set (all known champions) and apply to the in-progress season(s).
_in_progress = _to_train_df[~_to_train_df['season'].isin(_completed_seasons)]
if not _in_progress.empty and not _eligible.empty:
    beta_full = _to_fit_logistic(_to_features(_eligible), _eligible['is_champion'].values.astype(float))
    cur = _in_progress.copy()
    cur['p_raw'] = _to_predict_logistic(_to_features(cur), beta_full)
    cur['p_norm'] = cur.groupby('ranking_id')['p_raw'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0.0)
    for _, r in cur.iterrows():
        _title_odds_cache[(r['ranking_id'], r['team'])] = float(r['p_norm'])

# Pre-2004 historical seasons: also predict with the full-history-trained
# model so the UI surfaces something coherent on older snapshots, even though
# they're outside the training cutoff. Predictions there are extrapolations
# - calibration is not guaranteed.
_pre_window = _to_train_df[_to_train_df['season'] < TITLE_TRAIN_FROM_SEASON]
if not _pre_window.empty:
    pre = _pre_window.copy()
    pre['p_raw'] = _to_predict_logistic(_to_features(pre), beta_full if not _in_progress.empty else
                                         _to_fit_logistic(_to_features(_eligible), _eligible['is_champion'].values.astype(float)))
    pre['p_norm'] = pre.groupby('ranking_id')['p_raw'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0.0)
    for _, r in pre.iterrows():
        _title_odds_cache[(r['ranking_id'], r['team'])] = float(r['p_norm'])

# Per-snapshot rank (1 = highest odds among alive teams)
_title_odds_rank_cache = {}
_to_pairs_by_rid = {}
for (rid, team), odds in _title_odds_cache.items():
    if odds is None or odds <= 0:
        continue
    _to_pairs_by_rid.setdefault(rid, []).append((team, odds))
for rid, pairs in _to_pairs_by_rid.items():
    pairs.sort(key=lambda x: -x[1])
    rank_map = {}
    prev_odds = None
    prev_rank = 0
    for i, (team, odds) in enumerate(pairs, start=1):
        if odds != prev_odds:
            prev_rank = i
            prev_odds = odds
        rank_map[team] = prev_rank
    _title_odds_rank_cache[rid] = rank_map

print(f"  Title odds cached for {len(_title_odds_cache):,} (snapshot, team) pairs")


def _title_odds_val(ranking_id, team):
    return _title_odds_cache.get((int(ranking_id), team))


def _title_odds_rk(ranking_id, team):
    rm = _title_odds_rank_cache.get(int(ranking_id))
    return rm.get(team) if rm else None


# ── 1. Current standings ─────────────────────────────────────────────────────
print("Writing current_standings.json...")
latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank').copy()
latest_date = str(latest['date'].iloc[0])

standings_data = {
    'updated': latest_date,
    'teams': [
        {
            'rank':            int(r['rank']),
            'team':            r['name'],
            'display_name':    display_name(r['name'], r['season']),
            'conference':      conference(r['name'], r['season']),
            'rating':          round(float(r['rating']), 3),
            **_od_fields(r),
            'title_odds':      _title_odds_val(r['ranking_id'], r['name']),
            'title_odds_rank': _title_odds_rk(r['ranking_id'], r['name']),
            'record':          clean(r['record']),
            'last_match':      era_aware_last_match(clean(r['last_game_result']) if _played(r['last_game_result']) else last_game_as_of(r['name'], str(r['date']), r['season']), r['season']),
            'finals_status':   int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
            'cup_status':      int(r['cup_status']) if 'cup_status' in r and not pd.isna(r['cup_status']) else 0,
        }
        for _, r in latest.iterrows()
    ],
}
with open('docs/data/current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))

# ── 2. GOAT tables (end-of-RS + end-of-playoffs) ─────────────────────────────
# Two lists, matching the SAKIC/GRIFFEY fleet pattern:
#   goat_rs.json - top 50 single-season ratings at end of regular season, all teams.
#   goat_ps.json - top 50 single-season ratings at end of playoffs, Finals participants only.
# Both gated to fully-complete seasons (a season is "complete" once a
# season_flag == 2 row exists for that season - i.e. the Finals have ended).
print("Writing goat_rs.json + goat_ps.json...")

# Short / disrupted seasons - flagged on GOAT/Standings/Champions/TeamSummary
# rows so the UI can tag them inline. Small samples bias ratings; the tag
# adds context without altering the model. Categories drive UI color:
#   'cancelled' (red)  - season or major portion never played
#   'labor'     (amber) - strike or lockout shortened a played season
#   'covid'     (yellow) - COVID-related disruption
SHORT_SEASONS = {
    1999: {
        'tag': 'lockout 50g',
        'category': 'labor',
        'note': "The 1998-99 season was shortened to 50 games by a lockout that ran from July 1998 to January 1999.",
    },
    2012: {
        'tag': 'lockout 66g',
        'category': 'labor',
        'note': "The 2011-12 season was shortened to 66 games by a lockout that ran from July 2011 to December 2011.",
    },
    2020: {
        'tag': 'COVID bubble',
        'category': 'covid',
        'note': "The 2019-20 regular season was halted in March 2020 with ~65-70 games played per team; the season resumed in a single-site bubble in Orlando in July 2020.",
    },
    2021: {
        'tag': 'COVID 72g',
        'category': 'covid',
        'note': "The 2020-21 season was shortened to 72 games and started two months late (Dec 22) due to ongoing COVID disruption.",
    },
}

completed_seasons = set(df.loc[df['season_flag'] == 2, 'season'].astype(int).unique())


def build_goat(flag, require_finalist, sort_col='rating'):
    rows = df[(df['season_flag'] == flag) &
              (df['season'].astype(int).isin(completed_seasons))].copy()
    if require_finalist:
        # Champions only - the PS GOAT is a greatest-CHAMPIONS list; dominant
        # non-winning seasons live on the RS GOAT. Length rounds down to the
        # nearest 10 (capped at 50) so it grows cleanly as titles accrue.
        rows = rows[rows['finals_status'].fillna(0) == 2]
        n = min(50, (len(rows) // 10) * 10)
    else:
        n = 50
    # Drop any rows missing the sort column (older snapshots that pre-date
    # the O/D port wouldn't have rating_o/rating_d populated).
    rows = rows[rows[sort_col].notna()]
    rows = rows.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)
    out = []
    for i, (_, r) in enumerate(rows.iterrows()):
        s = int(r['season'])
        reg = _reg_record_lookup.get((r['name'], s), '')
        full = _full_record_lookup.get((r['name'], s), '')
        out.append({
            'rank':             i + 1,
            'team':             r['name'],
            'display_name':     display_name(r['name'], r['season']),
            'conference':       conference(r['name'], r['season']),
            'season':           s,
            'short_season':          s in SHORT_SEASONS,
            'short_season_tag':      SHORT_SEASONS.get(s, {}).get('tag', '')      if s in SHORT_SEASONS else '',
            'short_season_category': SHORT_SEASONS.get(s, {}).get('category', '') if s in SHORT_SEASONS else '',
            'short_season_note':     SHORT_SEASONS.get(s, {}).get('note', '')     if s in SHORT_SEASONS else '',
            'rating':           round(float(r['rating']), 3),
            **_od_fields(r),
            'record':           clean(full or r['record']),
            'regular_record':   reg,
            'playoff_record':   playoff_record(full, reg) if full else '',
            'finals_status':    int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
            'cup_status':       int(r['cup_status']) if 'cup_status' in r and not pd.isna(r['cup_status']) else 0,
        })
    return out


# Six GOAT files: {Rating, Offense, Defense} × {RS-end, PS-end}. The
# PS-end variants are restricted to Finals participants so the list shows
# actual championship contenders, not playoff flameouts. Mirrors DILLON.
goat_files = [
    ('goat_rs.json',   1, False, 'rating'),
    ('goat_ps.json',   2, True,  'rating'),
    ('goat_rs_o.json', 1, False, 'rating_o'),
    ('goat_rs_d.json', 1, False, 'rating_d'),
    ('goat_ps_o.json', 2, True,  'rating_o'),
    ('goat_ps_d.json', 2, True,  'rating_d'),
]
for fname, flag, require_finalist, sort_col in goat_files:
    payload = build_goat(flag=flag, require_finalist=require_finalist, sort_col=sort_col)
    with open(f'docs/data/{fname}', 'w') as f:
        json.dump(payload, f, separators=(',', ':'))

# ── 3. Per-team JSON files ───────────────────────────────────────────────────
print("Writing per-team JSON files...")
team_data = df[(df['is_game_day'] == 1) | (df['is_end_of_season'] == 1)].copy()
team_data = team_data.sort_values(['name', 'season', 'date'])

all_teams = sorted(df['name'].unique())
teams_index = []

for team in all_teams:
    tdf = team_data[team_data['name'] == team]
    if len(tdf) == 0:
        continue

    team_slug = slug(team)
    teams_index.append({
        'name': team,
        'display_name': current_display_name(team),
        'historical_names': historical_display_names(team),
        'conference': conference(team),
        'slug': team_slug,
    })

    seasons = {}
    for season, sdf in tdf.groupby('season'):
        rs_end = _rs_end_dates.get(season)
        final_reg = _reg_record_lookup.get((team, int(season)))
        entries = []
        for _, r in sdf.sort_values('date').iterrows():
            in_postseason = (rs_end is not None) and (r['date'] > rs_end) and (final_reg is not None)
            if in_postseason:
                reg = final_reg
                po  = playoff_record(r['record'], final_reg)
            else:
                reg = clean(r['record'])
                po  = ''
            entries.append({
                'date':              str(r['date']),
                'display_name':      display_name(team, season),
                'conference':        conference(team, season),
                'rating':            round(float(r['rating']), 3),
                'rank':              int(r['rank']),
                **_od_fields(r),
                'title_odds':        _title_odds_val(r['ranking_id'], team),
                'title_odds_rank':   _title_odds_rk(r['ranking_id'], team),
                'record':            clean(r['record']),
                'regular_record':    reg,
                'playoff_record':    po,
                'last_match':        era_aware_last_match(clean(r['last_game_result']) if _played(r['last_game_result']) else last_game_as_of(team, str(r['date']), season), season),
                'is_end_of_season':  int(r['is_end_of_season']),
                'season_flag':       int(r['season_flag']),
                'is_playoff':        int(is_playoff(season, r['date'])),
                'finals_status':     int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
                'cup_status':        int(r['cup_status']) if 'cup_status' in r and not pd.isna(r['cup_status']) else 0,
            })
        seasons[int(season)] = entries

    with open(f'docs/data/teams/{team_slug}.json', 'w') as f:
        json.dump({'team': team, 'conference': conference(team), 'seasons': seasons},
                  f, separators=(',', ':'))

teams_index.sort(key=lambda x: x['name'])
with open('docs/data/teams_index.json', 'w') as f:
    json.dump(teams_index, f, separators=(',', ':'))

# ── 4. Season standings files ─────────────────────────────────────────────────
print("Writing season standings files...")
all_seasons = sorted(df['season'].unique())

for season in all_seasons:
    sdf = df[df['season'] == season]
    snapshots = []
    for ranking_id, rdf in sdf.groupby('ranking_id'):
        rdf = rdf.sort_values('rank')
        snap_date = str(rdf['date'].iloc[0])
        flag = int(rdf['season_flag'].iloc[0])
        label = None
        if flag == 1:
            label = 'End of regular season'
        elif flag == 2:
            label = 'End of playoffs'

        snap_date_obj = rdf['date'].iloc[0]
        rs_end = _rs_end_dates.get(season)
        in_postseason = (rs_end is not None) and (snap_date_obj > rs_end)

        teams_snap = []
        for _, r in rdf.iterrows():
            if in_postseason:
                reg = _reg_record_lookup.get((r['name'], int(season)), r['record'])
                po  = playoff_record(r['record'], reg)
            else:
                reg = clean(r['record'])
                po  = ''
            played_today = _played(r['last_game_result'])
            teams_snap.append({
                'rank':            int(r['rank']),
                'team':            r['name'],
                'display_name':    display_name(r['name'], season),
                'conference':      conference(r['name'], season),
                'rating':          round(float(r['rating']), 3),
                **_od_fields(r),
                'title_odds':      _title_odds_val(r['ranking_id'], r['name']),
                'title_odds_rank': _title_odds_rk(r['ranking_id'], r['name']),
                'record':          clean(r['record']),
                'regular_record':  reg,
                'playoff_record':  po,
                'last_match':      era_aware_last_match(clean(r['last_game_result']) if played_today else last_game_as_of(r['name'], snap_date, season), season),
                'last_match_date': snap_date if played_today else last_game_date_as_of(r['name'], snap_date, season),
                'finals_status':   int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
            'cup_status':      int(r['cup_status']) if 'cup_status' in r and not pd.isna(r['cup_status']) else 0,
            })
        snapshots.append({'date': snap_date, 'label': label, 'teams': teams_snap})

    snapshots.sort(key=lambda x: x['date'])
    with open(f'docs/data/seasons/{int(season)}.json', 'w') as f:
        json.dump({'season': int(season), 'snapshots': snapshots}, f, separators=(',', ':'))

seasons_meta = {
    'seasons':    [int(s) for s in reversed(all_seasons)],
    'first_date': str(games['date_game'].min()),  # actual first game (not first rated date)
    'last_date':  str(games['date_game'].max()),
    'generated_at': datetime.now(timezone.utc).isoformat(),
    # Fleet-wide disrupted-season lookup - SPA references this to render
    # tags + footnotes consistently across Standings / Team Summary / Champions / GOAT.
    'disrupted_seasons': {
        str(year): {'tag': info['tag'], 'category': info['category'], 'note': info['note']}
        for year, info in SHORT_SEASONS.items()
    },
}
with open('docs/data/seasons_index.json', 'w') as f:
    json.dump(seasons_meta, f, separators=(',', ':'))

# ── 5. Champions table ────────────────────────────────────────────────────────
print("Writing champions.json...")

# Pre-Finals snapshot per season: the ranking_id of the last rating snapshot
# STRICTLY BEFORE the Finals series begins, for each season. Used in the
# Lists sub-view to evaluate matchup quality / closeness / upsets without
# the circularity of letting the Finals result colour the "going-in" rating.
# Mirrors DILLON's week-103 pre-SB snapshot pattern.
def _build_pre_finals_lookup():
    out = {}
    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        champ_names = season_df[season_df['champ'] == 1]['name'].unique()
        ru_names    = season_df[season_df['runnerup'] == 1]['name'].unique()
        if len(champ_names) == 0 or len(ru_names) == 0:
            continue
        champ, ru = champ_names[0], ru_names[0]
        rs_end = _rs_end_dates.get(season)
        season_games = games[games['season'] == season]
        if rs_end is not None:
            season_games = season_games[season_games['date_game'] > rs_end]
        # Finals = playoff games where these two specific teams meet
        finals = season_games[
            ((season_games['home_team_name'] == champ) & (season_games['visitor_team_name'] == ru)) |
            ((season_games['home_team_name'] == ru) & (season_games['visitor_team_name'] == champ))
        ]
        if finals.empty:
            continue
        finals_g1_date = finals['date_game'].min()
        # Latest ranking_id with date strictly before Finals Game 1.
        pre = season_df[season_df['date'] < finals_g1_date]
        if pre.empty:
            continue
        pre_id = int(pre['ranking_id'].max())
        snap = season_df[season_df['ranking_id'] == pre_id]
        for _, r in snap.iterrows():
            out[(r['name'], int(season))] = {
                'rating': round(float(r['rating']), 3),
                'rank':   int(r['rank']),
                'record': clean(r['record']),
                **_od_fields(r),
            }
    return out

_pre_finals_lookup = _build_pre_finals_lookup()
print(f"  pre-Finals snapshots computed for {len(set(s for (_, s) in _pre_finals_lookup))} seasons")

def pre_finals_fields(name, season, reg_record):
    """Return the pre-Finals rating/rank/playoff_record block, or empty if missing."""
    p = _pre_finals_lookup.get((name, int(season)))
    if p is None:
        return {}
    return {
        'rating_pre':         p['rating'],
        'rank_pre':           p['rank'],
        'rating_o_pre':       p.get('rating_o'),
        'rating_d_pre':       p.get('rating_d'),
        'rank_o_pre':         p.get('rank_o'),
        'rank_d_pre':         p.get('rank_d'),
        'playoff_record_pre': playoff_record(p['record'], reg_record),
    }


champions = []
for season in sorted(df['season'].unique(), reverse=True):
    sdf = df[(df['season'] == season) & (df['season_flag'] == 2)]
    if sdf.empty:
        continue
    champ_row = sdf[sdf['champ'] == 1]
    ru_row = sdf[sdf['runnerup'] == 1]
    if champ_row.empty or ru_row.empty:
        continue

    cr = champ_row.iloc[0]
    rr = ru_row.iloc[0]

    season_games = games[games['season'] == season]
    final_score = ''
    series_score = ''
    if not season_games.empty:
        last_game = season_games.sort_values('date_game').iloc[-1]
        if last_game['home_team_name'] == cr['name']:
            final_score = f"{int(last_game['home_pts'])}-{int(last_game['visitor_pts'])}"
        elif last_game['visitor_team_name'] == cr['name']:
            final_score = f"{int(last_game['visitor_pts'])}-{int(last_game['home_pts'])}"

        # Series: count champion vs runner-up wins in the postseason
        rs_end = _rs_end_dates.get(season)
        playoff_games = season_games[season_games['date_game'] > rs_end] if rs_end is not None else season_games
        finals = playoff_games[
            ((playoff_games['home_team_name'] == cr['name']) & (playoff_games['visitor_team_name'] == rr['name'])) |
            ((playoff_games['home_team_name'] == rr['name']) & (playoff_games['visitor_team_name'] == cr['name']))
        ]
        cw, rw = 0, 0
        for _, g in finals.iterrows():
            home_won = g['home_pts'] > g['visitor_pts']
            champ_was_home = g['home_team_name'] == cr['name']
            if home_won == champ_was_home:
                cw += 1
            else:
                rw += 1
        if cw + rw > 0:
            series_score = f"{cw}-{rw}"

    champ_reg = _reg_record_lookup.get((cr['name'], int(season)), '')
    ru_reg    = _reg_record_lookup.get((rr['name'], int(season)), '')

    champions.append({
        'season':       int(season),
        'series':       series_score,
        'final_score':  final_score,
        'champion': {
            'team':           cr['name'],
            'display_name':   display_name(cr['name'], season),
            'conference':     conference(cr['name'], season),
            'rating':         round(float(cr['rating']), 3),
            'rank':           int(cr['rank']),
            **_od_fields(cr),
            'record':         clean(cr['record']),
            'regular_record': champ_reg,
            'playoff_record': playoff_record(cr['record'], champ_reg),
            'cup_status':     int(cr['cup_status']) if 'cup_status' in cr and not pd.isna(cr['cup_status']) else 0,
            **pre_finals_fields(cr['name'], season, champ_reg),
        },
        'runner_up': {
            'team':           rr['name'],
            'display_name':   display_name(rr['name'], season),
            'conference':     conference(rr['name'], season),
            'rating':         round(float(rr['rating']), 3),
            'rank':           int(rr['rank']),
            **_od_fields(rr),
            'record':         clean(rr['record']),
            'regular_record': ru_reg,
            'playoff_record': playoff_record(rr['record'], ru_reg),
            'cup_status':     int(rr['cup_status']) if 'cup_status' in rr and not pd.isna(rr['cup_status']) else 0,
            **pre_finals_fields(rr['name'], season, ru_reg),
        },
    })

# Pre-1977 NBA Finals counts (1947-1976), keyed by team name as it appears in our data.
# Franchises that no longer exist or relocated under different names (e.g. Minneapolis
# Lakers, Syracuse Nationals, St. Louis Hawks, Fort Wayne Pistons) are NOT carried over -
# matches the city-name-separate philosophy used elsewhere in the site.
# The 1977-1979 champions (Blazers, Bullets, Sonics) are now tracked by the live ratings
# era and excluded here.
PRE_1977_CHAMPIONSHIPS = {
    'Boston Celtics':         13,  # 1957, 1959-66, 1968, 1969, 1974, 1976
    'Los Angeles Lakers':      1,  # 1972
    'Philadelphia 76ers':      1,  # 1967
    'New York Knicks':         2,  # 1970, 1973
    'Milwaukee Bucks':         1,  # 1971
    'Golden State Warriors':   1,  # 1975
}

PRE_1977_RUNNER_UPS = {
    'New York Knicks':         4,  # 1951, 1952, 1953, 1972
    'Boston Celtics':          1,  # 1958
    'Los Angeles Lakers':      8,  # 1962, 1963, 1965, 1966, 1968, 1969, 1970, 1973
    'Milwaukee Bucks':         1,  # 1974
    'Washington Wizards':      1,  # 1975 (as Bullets)
    'Phoenix Suns':            1,  # 1976
}

# Running counts: walk chronologically (oldest first), seeded with pre-1977 totals
_champ_count = dict(PRE_1977_CHAMPIONSHIPS)
_ru_count    = dict(PRE_1977_RUNNER_UPS)
for entry in reversed(champions):
    ct = entry['champion']['team']
    rt = entry['runner_up']['team']
    _champ_count[ct] = _champ_count.get(ct, 0) + 1
    _ru_count[rt]    = _ru_count.get(rt, 0) + 1
    entry['champion']['title_count']      = _champ_count[ct]
    entry['runner_up']['runner_up_count'] = _ru_count[rt]

with open('docs/data/champions.json', 'w') as f:
    json.dump({'NBA': champions}, f, separators=(',', ':'))

print(f"Done. {len(teams_index)} teams, {len(standings_data['teams'])} in current standings.")
print(f"Wrote {len(all_seasons)} season files. Standings date: {latest_date}")

# Hygiene: flag any rated team missing from TEAM_CONFERENCE. Without this,
# expansion teams (or future renames) silently fall through to 'Other' and
# disappear from the conference filter pillbox.
_unknown = sorted({t for t in df['name'].unique() if t not in TEAM_CONFERENCE})
if _unknown:
    print()
    print('⚠️  WARNING: teams in rated data missing from TEAM_CONFERENCE:')
    for t in _unknown:
        print(f'    - {t!r}')
    print('    These teams will display as "Other" until added.')
    print()
