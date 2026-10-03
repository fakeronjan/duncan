"""Real NBA playoff seeds for the current season -> nba_playoff_seeds.json
(read by playoff_sim.REAL_SEEDS). Runs daily; does nothing until the regular
season is over.

Source: ESPN's standings (playoffSeed, 1-10 per conference) as of the end
of the regular season. ESPN rewrites 7-10 after the play-in, so once the
play-in games are played, 7-10 come from them: the 7 hosts the 8, the 9
hosts the 10 (the game whose loser plays again is 7 v 8).
"""
# ── Shared core (same in every fleet site's playoff_seeds.py) ─────────────────
# Once a regular season is over, the playoff sims seed from the real seeds
# (playoff_sim.REAL_SEEDS), never their own tiebreak estimate. run() fetches
# them, checks them and writes them; seeds missing or failing a check give a
# warning for GRACE_DAYS after the regular season, then the run fails.
#
# Checks: every seed filled once and every team known; within each seeding
# tier, seeds follow the standings (a source can only differ from our own
# order where records are level); and once real playoff games exist, every
# series between two seeded teams of the same group was opened at the
# better seed.
import json
import os
import re
import urllib.parse
import urllib.request

import pandas as pd

GRACE_DAYS = 2
# Wikipedia asks for a descriptive agent with contact details; ESPN and the
# league APIs refuse agents with an email in them, so they get a plain one.
_UA_WIKI = {'User-Agent': 'fakeronjan-sports/1.0 (rjsikdar@gmail.com)'}
_UA = {'User-Agent': 'Mozilla/5.0'}


def get_json(url):
    ua = _UA_WIKI if 'wikipedia.org' in url else _UA
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=30))


def wikitext(title):
    d = get_json('https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(
        {'action': 'parse', 'page': title, 'prop': 'wikitext', 'format': 'json', 'formatversion': 2,
         'redirects': 1}))
    return d['parse']['wikitext'] if 'parse' in d else None


def bracket_links(text):
    """{linked page name: seed label} from a Wikipedia bracket template's
    RDn-seedXX / RDn-teamXX pairs (first label seen per team)."""
    seeds = {}
    sd = {(m.group(1), int(m.group(2))): m.group(3)
          for m in re.finditer(r'\|\s*RD(\d+)-seed0*(\d+)\s*=\s*([^\n|]*)', text)}
    for m in re.finditer(r'\|\s*RD(\d+)-team0*(\d+)\s*=\s*([^\n]*)', text):
        lab = re.sub(r'[^\w]', '', sd.get((m.group(1), int(m.group(2))), ''))
        link = re.search(r'\[\[([^\]|]+)', m.group(3))
        if not lab or not link:
            continue
        name = re.sub(r'^\d{4}(?:[–-]\d{2,4})?\s+', '', link.group(1).strip())
        seeds.setdefault(re.sub(r'\s+season$', '', name), lab)
    return seeds


def _problem(seeds, teams, tiers, rec, ps_games):
    """seeds: {group: [team, ...]} best first. tiers: [(group, [seed numbers])].
    rec: {team: standings value, higher = better}. ps_games: [(home, away,
    ...)] in date order. Returns a description of the first failed check, or None."""
    seen = [t for lst in seeds.values() for t in lst]
    if any(t not in teams for t in seen) or len(set(seen)) != len(seen):
        return f"unknown or repeated teams: {seen}"
    for g, nums in tiers:
        lst = seeds.get(g, [])
        if len(lst) < max(nums):
            return f"{g} has {len(lst)} seeds, expected {max(nums)}"
        vals = [rec.get(lst[k - 1], 0) for k in nums]
        if any(a < b - 1e-9 for a, b in zip(vals, vals[1:])):
            return f"{g} seeds {nums} don't follow the standings: {[lst[k - 1] for k in nums]} {vals}"
    rank = {t: (g, k) for g, lst in seeds.items() for k, t in enumerate(lst)}
    first = {}
    for x in ps_games:
        first.setdefault(frozenset(x[:2]), x[0])
    for pair, host in first.items():
        x, y = sorted(pair)
        if x in rank and y in rank and rank[x][0] == rank[y][0]:
            if host != min(pair, key=lambda t: rank[t][1]):
                return f"{' vs '.join(pair)} opened at {host}, not the better seed"
    return None


def run(path, season, rs_end, fetch, teams, tiers, rec, ps_games, today=None, label=''):
    """rs_end: last regular-season date (None while it's still going).
    fetch(stored) -> seeds dict (may refine stored ones, e.g. from play-in
    games) or None when the source doesn't have them yet."""
    if rs_end is None:
        return None
    data = json.load(open(path)) if os.path.exists(path) else {}
    stored = data.get(str(season))
    problem = None
    try:
        seeds = fetch(stored)
        if seeds is None:
            problem = 'the source has no seeds yet'
        else:
            problem = _problem(seeds, teams, tiers, rec, ps_games)
    except Exception as e:                       # network, parsing
        seeds, problem = None, f'{type(e).__name__}: {e}'
    if problem is None:
        if seeds != stored:
            data[str(season)] = seeds
            json.dump(dict(sorted(data.items())), open(path, 'w'), indent=1)
            print(f"  {season} {label}playoff seeds -> {path}: {seeds}")
        return seeds
    if stored is not None and _problem(stored, teams, tiers, rec, ps_games) is None:
        print(f"::warning::{season} {label}seed refresh failed ({problem}); keeping the stored seeds")
        return stored
    days = ((today or pd.Timestamp.now()).normalize() - pd.Timestamp(rs_end).normalize()).days
    msg = f"{season} {label}playoff seeds not usable yet: {problem}"
    if days > GRACE_DAYS:
        raise RuntimeError(msg + f" ({days} days after the regular season)")
    print(f"::warning::{msg}")
    return None


# ── NBA ──────────────────────────────────────────────────────────────────────
from duncan import REGULAR_SEASON_GAMES

SEEDS_JSON = 'nba_playoff_seeds.json'
ESPN_URL = 'https://site.api.espn.com/apis/v2/sports/basketball/nba/standings?season={}'


def season_state(games_csv='all_nba_games.csv'):
    """(season, rs_end or None, teams, win% by team, postseason games as
    (home, away, winner) in date order)."""
    g = pd.read_csv(games_csv)
    season = int(g['season'].max())
    g = g[(g['season'] == season) & g['home_pts'].notna()]
    if 'is_nba_cup_final' in g.columns:
        g = g[g['is_nba_cup_final'] != 1]                 # not a standings game
    g = g.assign(date=pd.to_datetime(g['date_game'], format='mixed')).sort_values('date', kind='stable')
    n = REGULAR_SEASON_GAMES.get(season, 82)
    counts, is_rs = {}, []
    for h, a in zip(g['home_team_name'], g['visitor_team_name']):
        counts[h] = counts.get(h, 0) + 1; counts[a] = counts.get(a, 0) + 1
        is_rs.append(counts[h] <= n and counts[a] <= n)
    g['is_rs'] = is_rs
    rs = g[g['is_rs']]
    teams = set(rs['home_team_name']) | set(rs['visitor_team_name'])
    played = pd.concat([rs['home_team_name'], rs['visitor_team_name']]).value_counts()
    rs_end = rs['date'].max() if len(teams) and (played >= n).all() else None
    wins = pd.concat([rs.loc[rs['home_pts'] > rs['visitor_pts'], 'home_team_name'],
                      rs.loc[rs['visitor_pts'] > rs['home_pts'], 'visitor_team_name']]).value_counts()
    rec = {t: wins.get(t, 0) / max(played.get(t, 1), 1) for t in teams}
    po = g[~g['is_rs']]
    ps = [(h, a, h if hp > vp else a) for h, a, hp, vp in
          po[['home_team_name', 'visitor_team_name', 'home_pts', 'visitor_pts']].itertuples(index=False)]
    return season, rs_end, teams, rec, ps


def _by_nickname(name, teams):
    nick = name.split()[-1]
    hit = [t for t in teams if t.split()[-1] == nick]
    if len(hit) != 1:
        raise ValueError(f"can't match {name!r} to one team: {hit}")
    return hit[0]


def espn_seeds(season, teams):
    d = get_json(ESPN_URL.format(season))
    out = {}
    for ch in d.get('children', []):
        conf = 'East' if 'East' in ch['name'] else 'West'
        rows = []
        for e in ch['standings']['entries']:
            st = {s['name']: s.get('value') for s in e['stats']}
            if st.get('playoffSeed') and st['playoffSeed'] <= 10:
                rows.append((int(st['playoffSeed']), _by_nickname(e['team']['displayName'], teams)))
        if sorted(k for k, _ in rows) != list(range(1, 11)):
            return None                                     # not seeded yet
        out[conf] = [t for _, t in sorted(rows)]
    return out if len(out) == 2 else None


def with_playin(seeds, ps):
    """Seeds 7-10 from the play-in games once each conference's first two
    are played: the game whose loser plays again is 7 v 8 (7 hosts), the
    other is 9 v 10 (9 hosts)."""
    seeds = {c: list(v) for c, v in seeds.items()}
    for c, lst in seeds.items():
        four = set(lst[6:10])
        games = [x for x in ps if x[0] in four and x[1] in four]
        if len(games) < 2:
            continue
        later = {t for x in games[2:] for t in x[:2]}
        first2 = games[:2]
        loser = lambda x: x[1] if x[2] == x[0] else x[0]
        seven_eight = [x for x in first2 if loser(x) in later]
        if len(seven_eight) != 1:
            continue                                   # 8th-place game not played yet
        nine_ten = [x for x in first2 if x is not seven_eight[0]][0]
        lst[6], lst[7] = seven_eight[0][0], seven_eight[0][1]
        lst[8], lst[9] = nine_ten[0], nine_ten[1]
    return seeds


def main(today=None):
    season, rs_end, teams, rec, ps = season_state()

    def fetch(stored):
        seeds = stored or espn_seeds(season, teams)
        return with_playin(seeds, ps) if seeds else None

    tiers = [(c, list(range(1, 11))) for c in ('East', 'West')]
    run(SEEDS_JSON, season, rs_end, fetch, teams, tiers, rec, ps, today=today, label='NBA ')


if __name__ == '__main__':
    main()
