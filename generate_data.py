"""
generate_data.py — reads duncan_ratings_with_standings.csv and writes JSON for the DUNCAN web frontend.
Run after duncan.py. Outputs to docs/data/.

Mirrors the LOBO/ZIDANE site architecture, with NBA-specific tweaks:
  - East/West conference mapping (per team, including historical relocations)
  - Single-year season display (e.g. "2025" = 2024-25 season per basketball-reference)
"""

import pandas as pd
import json
import os
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


def conference(team):
    return TEAM_CONFERENCE.get(team, 'Other')


def clean(val):
    if pd.isna(val):
        return ''
    return str(val)


def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')


# is_game_day: any row where the team actually played that snapshot date
df['is_game_day'] = (df['last_game_result'] != 'No Game').astype(int)
# is_end_of_season: collapse season_flag (1=last regular, 2=last postseason) to one boolean
df['is_end_of_season'] = df['season_flag'].isin([1, 2]).astype(int)

# Per-team forward-filled last game (so EOS rows that aren't game days still show prior game)
_last_game_history = {}
for team, tdf in df[df['is_game_day'] == 1].sort_values('date').groupby('name'):
    _last_game_history[team] = (
        [str(d) for d in tdf['date'].tolist()],
        tdf['last_game_result'].tolist(),
    )


def last_game_as_of(team, snap_date_str):
    entry = _last_game_history.get(team)
    if not entry:
        return ''
    dates, games_list = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return games_list[idx] if idx >= 0 else ''


def last_game_date_as_of(team, snap_date_str):
    entry = _last_game_history.get(team)
    if not entry:
        return ''
    dates, _ = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return dates[idx] if idx >= 0 else ''


# Per-season last regular-season date — used to flag playoff vs regular-season entries
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
            'conference':      conference(r['name']),
            'rating':          round(float(r['rating']), 3),
            'record':          clean(r['record']),
            'last_match':      clean(r['last_game_result']) if r['last_game_result'] != 'No Game' else last_game_as_of(r['name'], str(r['date'])),
            'finals_status':   int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
        }
        for _, r in latest.iterrows()
    ],
}
with open('docs/data/current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))

# ── 2. GOAT table ─────────────────────────────────────────────────────────────
# Only include fully-complete seasons (flag=2 = Finals ended). Excludes
# in-progress seasons whose regular season is done but Finals haven't started/finished.
print("Writing goat_teams.json...")
eos_all = df[df['season_flag'] == 2].copy()
eos_top = eos_all.sort_values('rating', ascending=False).head(50).reset_index(drop=True)

goat_data = []
for i, (_, r) in enumerate(eos_top.iterrows()):
    reg = _reg_record_lookup.get((r['name'], int(r['season'])), '')
    goat_data.append({
        'rank':           i + 1,
        'team':           r['name'],
        'conference':     conference(r['name']),
        'season':         int(r['season']),
        'rating':         round(float(r['rating']), 3),
        'record':         clean(r['record']),
        'regular_record': reg,
        'playoff_record': playoff_record(r['record'], reg),
        'finals_status':  int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
    })
with open('docs/data/goat_teams.json', 'w') as f:
    json.dump(goat_data, f, separators=(',', ':'))

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
    teams_index.append({'name': team, 'conference': conference(team), 'slug': team_slug})

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
                'rating':            round(float(r['rating']), 3),
                'rank':              int(r['rank']),
                'record':            clean(r['record']),
                'regular_record':    reg,
                'playoff_record':    po,
                'last_match':        clean(r['last_game_result']) if r['last_game_result'] != 'No Game' else last_game_as_of(team, str(r['date'])),
                'is_end_of_season':  int(r['is_end_of_season']),
                'season_flag':       int(r['season_flag']),
                'is_playoff':        int(is_playoff(season, r['date'])),
                'finals_status':     int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
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
            label = 'End of season (Finals)'

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
            played_today = r['last_game_result'] != 'No Game'
            teams_snap.append({
                'rank':            int(r['rank']),
                'team':            r['name'],
                'conference':      conference(r['name']),
                'rating':          round(float(r['rating']), 3),
                'record':          clean(r['record']),
                'regular_record':  reg,
                'playoff_record':  po,
                'last_match':      clean(r['last_game_result']) if played_today else last_game_as_of(r['name'], snap_date),
                'last_match_date': snap_date if played_today else last_game_date_as_of(r['name'], snap_date),
                'finals_status':   int(r['finals_status']) if not pd.isna(r['finals_status']) else 0,
            })
        snapshots.append({'date': snap_date, 'label': label, 'teams': teams_snap})

    snapshots.sort(key=lambda x: x['date'])
    with open(f'docs/data/seasons/{int(season)}.json', 'w') as f:
        json.dump({'season': int(season), 'snapshots': snapshots}, f, separators=(',', ':'))

seasons_meta = {
    'seasons':    [int(s) for s in reversed(all_seasons)],
    'first_date': str(df['date'].min()),
    'last_date':  str(df['date'].max()),
}
with open('docs/data/seasons_index.json', 'w') as f:
    json.dump(seasons_meta, f, separators=(',', ':'))

# ── 5. Champions table ────────────────────────────────────────────────────────
print("Writing champions.json...")

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
        'final_score':  final_score,
        'series_score': series_score,
        'champion': {
            'team':           cr['name'],
            'conference':     conference(cr['name']),
            'rating':         round(float(cr['rating']), 3),
            'rank':           int(cr['rank']),
            'record':         clean(cr['record']),
            'regular_record': champ_reg,
            'playoff_record': playoff_record(cr['record'], champ_reg),
        },
        'runner_up': {
            'team':           rr['name'],
            'conference':     conference(rr['name']),
            'rating':         round(float(rr['rating']), 3),
            'rank':           int(rr['rank']),
            'record':         clean(rr['record']),
            'regular_record': ru_reg,
            'playoff_record': playoff_record(rr['record'], ru_reg),
        },
    })

# Running counts: walk chronologically (oldest first)
_champ_count = {}
_ru_count = {}
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
