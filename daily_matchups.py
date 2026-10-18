"""Daily Matchups: every game of a day, previewed from the ratings going into
it - win probability, DUNCAN's line and O/U (projected total), and the stakes
(each team's playoff and title odds with a win vs with a loss). After the
games: the final score and whether DUNCAN's pick was right. The NBA port of
DILLON's Weekly Matchups (dillon/weekly_matchups.py).

Stakes come from the season sim (playoff_sim.py): one run of 10k simulations
from the last snapshot before the day, split by each game's simulated result.
A day with no snapshot yet this season (opening night) previews from last
season's final ratings. Playoff games use the series' next game in the sim.
The NBA Cup final doesn't count in the standings: Quality only, no stakes.

Line and O/U (fit on 1978-2026 games, leave-one-season-out; line MAE 9.5 vs
11.1 pick'em, O/U MAE 14.4 vs 16.1 league average only):
    margin = LINE_LAM * (home rating - away rating + home edge)
        home edge = the sim's era home-court value (0 at neutral sites), so
        the line and the win probability always name the same favorite
    total  = 2 * mu + TOTAL_B * (both offenses - both defenses)
        mu = league points per team-game this season so far, shrunk toward
        last season's with MU_PRIOR_GAMES pseudo team-games
"""
import hashlib
import multiprocessing as _mp
import os
import pickle
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import ndtr

import playoff_sim

N_SIMS = 10_000                   # fleet standard since 2026-10-02
LINE_LAM = 0.905
TOTAL_B = 0.835
MU_PRIOR_GAMES = 400
ET = ZoneInfo('America/New_York')


def _record(w, l):
    return f"{w}-{l}"


def _kickoff(date, start):
    """basketball-reference start times are Eastern ('7:30p') -> UTC ISO."""
    if not isinstance(start, str) or not start.strip():
        return None
    s = start.strip().lower()
    try:
        hh, mm = s[:-1].split(':')
        h = int(hh) % 12 + (12 if s.endswith('p') else 0)
        local = pd.Timestamp(date).to_pydatetime().replace(hour=h, minute=int(mm), tzinfo=ET)
    except (ValueError, IndexError):
        return None
    return pd.Timestamp(local).tz_convert('UTC').strftime('%Y-%m-%dT%H:%MZ')


def _win_value(y, wins, ctrl):
    """(mean of y, change in y per extra win) for outcome y (per-sim bool)
    and the team's final regular-season wins in each sim.

    Splitting the sims by this one game's result (as DILLON does) also sorts
    them by how strong the drift draw made each team, which inflated early-
    season stakes ~3.5x over late-season ones (2010-26 Stakes score 72 for
    the first 10% of days vs 20 for the last 10%). Holding the drift fixed
    removes that, but one game's own result is then too noisy for title odds
    at 10k sims (opening-night swings came out negative). So: regress y on
    the team's final win total plus both teams' drift offsets. Given the
    drift, the win total varies only by luck across all remaining games, so
    the slope is the value of one more win, with much less noise."""
    y = y.astype(float)
    m = float(y.mean())
    X = np.column_stack([np.ones(len(y)), wins] + ([ctrl] if ctrl is not None else []))
    if np.ptp(wins) == 0:
        return m, 0.0
    return m, max(float(np.linalg.lstsq(X, y, rcond=None)[0][1]), 0.0)


def _raw_split(y, won):
    """Playoff games: no drift left, so split the sims by the game's result."""
    y = y.astype(float)
    return (float(y[won].mean()) if won.any() else None,
            float(y[~won].mean()) if (~won).any() else None)


def _around(m, b, pw):
    """Split mean m into (with a win, with a loss) for a +b per-win effect."""
    clip = lambda v: min(max(v, 0.0), 1.0)
    return clip(m + b * (1 - pw)), clip(m - b * pw)


def _mu_by_day(season_games, prev_mu):
    """League points per team-game going into each day: this season so far,
    shrunk toward last season's mean."""
    played = season_games[season_games['home_pts'].notna()]
    per_day = played.groupby('date').apply(
        lambda x: (x['home_pts'].sum() + x['visitor_pts'].sum(), 2 * len(x)))
    out, s, n = {}, 0.0, 0
    for d in sorted(set(season_games['date'])):
        out[d] = (s + MU_PRIOR_GAMES * prev_mu) / (n + MU_PRIOR_GAMES)
        if d in per_day.index:
            s += per_day[d][0]; n += per_day[d][1]
    return out


def build_season(season, sim_games, ratings, rs_games, conf_of, div_of, rs_end,
                 cup_finals, start_times, prev_mu, n_sims=N_SIMS, log=print):
    """sim_games: the season's games as the title-odds sim takes them (season,
    date, home, away, home_pts, visitor_pts, is_neutral; NaN pts =
    scheduled). ratings: (season, date, name, rating, rank, rating_o,
    rating_d) for this season and the last. cup_finals: NBA Cup final rows
    (date, home, away, home_pts, visitor_pts). start_times: {(date, home,
    away): '7:30p'}. Returns the season's days (list of dicts)."""
    g = sim_games[sim_games['season'] == season].sort_values(
        ['date', 'home', 'away'], kind='stable').reset_index(drop=True)
    snaps = {d: x.set_index('name') for d, x in ratings.groupby('date')}
    cur = sorted(d for d in snaps if ratings.loc[ratings['date'] == d, 'season'].iloc[0] == season)
    prev = sorted(d for d in snaps if d not in set(cur) and d < (cur[0] if cur else pd.Timestamp.max))
    # New teams (expansion / relocation): ratings blended toward a starting
    # point over their first games, as the title-odds sim does.
    priors = playoff_sim.new_team_priors(season, ratings)
    played_g = g[g['home_pts'].notna()]

    def steady(snap, d):
        snap = snap.copy()
        for t, prior in priors.items():
            n = int(((played_g['home'] == t) | (played_g['away'] == t))[played_g['date'] <= d].sum())
            w = playoff_sim.new_team_weight(n)
            if t in snap.index:
                raw = snap.loc[t, ['rating', 'rating_o', 'rating_d']].astype(float)
                snap.loc[t, 'rating'] = w * raw['rating'] + (1 - w) * prior
                snap.loc[t, 'rating_o'] = w * raw['rating_o'] + (1 - w) * prior / 2
                snap.loc[t, 'rating_d'] = w * raw['rating_d'] + (1 - w) * prior / 2
            else:                                     # not rated yet (opening-night preview)
                snap.loc[t, ['rating', 'rating_o', 'rating_d', 'rank']] = [prior, prior / 2, prior / 2, np.nan]
        return snap
    cur_set = set(cur)
    snaps = {d: (steady(x, d) if d in cur_set else x) for d, x in snaps.items()}
    sim_ratings = {d: dict(zip(snaps[d].index, snaps[d]['rating'])) for d in cur}
    sim = playoff_sim.SeasonSim(season, g, rs_games, conf_of, div_of, sim_ratings, rs_end)
    rs_keys = set(zip(sim.rs['date'], sim.rs['home'], sim.rs['away']))
    mu = _mu_by_day(g, prev_mu)

    cup = cup_finals.copy()
    days = sorted(set(g['date']) | set(cup['date']))
    unplayed_days = sorted(set(g.loc[g['home_pts'].isna(), 'date']))
    preview_day = unplayed_days[0] if unplayed_days else None

    days_out, previewed = [], False
    for day in days:
        if previewed:
            break                                     # live season: preview only the next game day
        before = [d for d in cur if d < day]
        if before:
            snap_d = before[-1]
        elif prev:
            snap_d = prev[-1]                         # opening night(s): last season's final ratings
            snaps[snap_d] = steady(snaps[snap_d], snap_d)   # + any new team at its starting point
            sim.ratings[snap_d] = dict(zip(snaps[snap_d].index, snaps[snap_d]['rating']))
        else:
            continue                                  # 1977: no ratings yet
        rt = snaps[snap_d]
        cap = {}
        sim.odds_at(snap_d, n_sims=n_sims, capture=cap)
        idx = {t: i for i, t in enumerate(cap['teams'])}
        made, champ = cap['made'], cap['champ']
        rest = cap.get('rest')
        E, Wsim = cap.get('E'), cap.get('W')
        value = {}                                    # team -> win values, once per day

        def win_value(t, opp):
            if t not in value:
                it = idx[t]
                ctrl = None if E is None else E[:, [it, idx[opp]]]
                value[t] = (_win_value(made[:, it], Wsim[:, it], ctrl),
                            _win_value(champ == it, Wsim[:, it], ctrl))
            return value[t]
        ps_next = {frozenset((a, b)): (a, won, bo) for a, b, won, bo in cap['ps_next']}

        # Regular-season records going into the day; series scores for playoff games.
        played = g[(g['date'] < day) & g['home_pts'].notna()]
        rec, series = {}, {}
        for x in played.itertuples(index=False):
            win = x.home if x.home_pts > x.visitor_pts else x.away
            if (x.date, x.home, x.away) in rs_keys:
                for t in (x.home, x.away):
                    w, l = rec.get(t, (0, 0))
                    rec[t] = (w + (t == win), l + (t != win))
            else:
                k = frozenset((x.home, x.away))
                series.setdefault(k, {}).setdefault(win, 0)
                series[k][win] += 1

        day_games = [(x, False) for x in g[g['date'] == day].itertuples(index=False)]
        day_games += [(x, True) for x in cup[cup['date'] == day].itertuples(index=False)]
        games_out = []
        for x, is_cup in day_games:
            h, a = x.home, x.away
            neutral = is_cup or int(getattr(x, 'is_neutral', 0) or 0) == 1 or (
                season == 2020 and day >= playoff_sim.BUBBLE_START)
            hp = 0.0 if neutral else sim.hp
            blank = pd.Series({'rating': 0.0, 'rating_o': 0.0, 'rating_d': 0.0, 'rank': np.nan})
            rh = rt.loc[h] if h in rt.index else blank
            ra = rt.loc[a] if a in rt.index else blank
            is_rs = (x.date, h, a) in rs_keys
            stakes, ser = None, None
            if is_cup:
                p_home = float(ndtr(sim.A * (rh['rating'] - ra['rating'] + hp)))
            elif is_rs:
                if rest is None or h not in idx or a not in idx:
                    continue
                j = rest.index[(rest['home'] == h) & (rest['away'] == a) & (rest['date'] == day)]
                if not len(j):
                    continue
                hw = cap['hw'][:, j[0]]
                p_home = float(hw.mean())
                stakes = {}
                for t, opp, pw in ((h, a, p_home), (a, h, 1 - p_home)):
                    (po_m, po_b), (ti_m, ti_b) = win_value(t, opp)
                    po_w, po_l = _around(po_m, po_b, pw)
                    ti_w, ti_l = _around(ti_m, ti_b, pw)
                    stakes['home' if t == h else 'away'] = {
                        'po_win': po_w, 'po_loss': po_l, 'title_win': ti_w, 'title_loss': ti_l}
            else:
                hit = ps_next.get(frozenset((h, a)))
                if hit is None:
                    continue
                ta, won_a, best_of = hit
                sw = series.get(frozenset((h, a)), {})
                if best_of > 2 and pd.isna(x.home_pts) and max(sw.values(), default=0) >= best_of // 2 + 1:
                    continue                      # an "if necessary" game after the series ended (2026 Finals Game 6)
                hw = won_a if ta == h else ~won_a
                p_home = float(hw.mean())
                stakes = {}
                for t, won in ((h, hw), (a, ~hw)):
                    it = idx[t]
                    ti_w, ti_l = _raw_split(champ == it, won)
                    # Play-in games decide who makes the playoffs proper.
                    po_w, po_l = _raw_split(made[:, it], won) if best_of <= 2 else (None, None)
                    stakes['home' if t == h else 'away'] = {
                        'po_win': po_w, 'po_loss': po_l, 'title_win': ti_w, 'title_loss': ti_l}
                # best_of 1 = a play-in game; 2 = 2020's 8-vs-9 play-in (the 9 needs two wins)
                ser = {'home_wins': sw.get(h, 0), 'away_wins': sw.get(a, 0), 'best_of': int(best_of)}
            margin = LINE_LAM * (rh['rating'] - ra['rating'] + hp)
            total = 2 * mu.get(day, prev_mu) + TOTAL_B * (
                (rh['rating_o'] + ra['rating_o']) - (rh['rating_d'] + ra['rating_d']))
            game = {
                'home': h, 'away': a, 'neutral': bool(neutral),
                'home_record': _record(*rec.get(h, (0, 0))), 'away_record': _record(*rec.get(a, (0, 0))),
                'home_rank': None if pd.isna(rh['rank']) else int(rh['rank']),
                'away_rank': None if pd.isna(ra['rank']) else int(ra['rank']),
                'p_home': round(p_home, 4),
                'line': round(float(margin) * 2) / 2,          # home by this many (negative = away)
                'total': round(float(total) * 2) / 2,          # DUNCAN O/U: projected combined points
                'stakes': None if stakes is None else
                          {k: {kk: (None if vv is None else round(vv, 4)) for kk, vv in v.items()} for k, v in stakes.items()},
                'kickoff': _kickoff(day, start_times.get((day, h, a))),
                'result': None,
            }
            if ser is not None:
                game['series'] = ser
            if is_cup:
                game['cup_final'] = True
            if not pd.isna(x.home_pts):
                game['result'] = {'home': int(x.home_pts), 'away': int(x.visitor_pts)}
            game['_q'] = (min(rh['rating'], ra['rating']), max(rh['rating'], ra['rating']))
            games_out.append(game)
        # The next game day with a game left on it: an unplayed "if
        # necessary" game after its series ended doesn't count.
        previewed = preview_day is not None and day >= preview_day and bool(games_out)
        if games_out:
            days_out.append({'date': day.strftime('%Y-%m-%d'),
                             'playoffs': any('series' in x for x in games_out), 'games': games_out})
        log(f'  {season} {day.date()}: {len(games_out)} games')
    return days_out


# ── Juice: Quality x Stakes ──────────────────────────────────────────────────
# As DILLON (weekly_matchups.py): Quality = 35% the worse team's rating, 35%
# the better team's, 30% closeness; Stakes = both teams' playoff-odds swing +
# K x their title-odds swing, K ramping 4 -> 8 over the regular season
# (playoff games: title swing only, K = 8). Each is ranked 0-100 and Juice =
# sqrt(Quality x Stakes).
#
# Unlike DILLON, the ranking pool is COMPLETED seasons only: ranking against
# the live season too would move every historical score every day and
# rewrite all 50 season files nightly. Live-season games are placed within
# the completed seasons' distribution instead, so history only re-ranks once
# a year, when a season finishes.
TITLE_K_START, TITLE_K_END = 4.0, 8.0


def _pct(ref_sorted, x):
    """Percentile (0-1) of x within ref, matching pandas rank(pct=True)'s
    average rank for values already in ref."""
    lo = np.searchsorted(ref_sorted, x, side='left')
    hi = np.searchsorted(ref_sorted, x, side='right')
    return ((lo + hi + 1) / 2) / len(ref_sorted)


def add_juice(seasons, pool_seasons):
    """seasons: {season: days (build_season output)}; pool_seasons: the
    completed seasons to rank against. Adds 'quality', 'stakes_score' and
    'juice' (0-100) to every game and drops the internal '_q'."""
    rows = []
    for season, days in seasons.items():
        rs_days = [d['date'] for d in days if not d['playoffs']]
        first, last = (rs_days[0], rs_days[-1]) if rs_days else (None, None)
        span = (pd.Timestamp(last) - pd.Timestamp(first)).days if rs_days else 1
        for d in days:
            if d['playoffs']:
                k = TITLE_K_END
            else:
                frac = (pd.Timestamp(d['date']) - pd.Timestamp(first)).days / max(span, 1)
                k = TITLE_K_START + (TITLE_K_END - TITLE_K_START) * min(frac, 1.0)
            for g in d['games']:
                qmin, qmax = g.pop('_q')
                close = 1 - abs(2 * g['p_home'] - 1)
                stake = None
                if g['stakes'] is not None:
                    st = list(g['stakes'].values())
                    po = sum((s['po_win'] or 0) - (s['po_loss'] or 0) for s in st)
                    ti = sum((s['title_win'] or 0) - (s['title_loss'] or 0) for s in st)
                    stake = po + k * ti
                rows.append((g, season in pool_seasons, qmin, qmax, close, stake))
    if not rows:
        return
    df = pd.DataFrame([r[1:] for r in rows], columns=['pool', 'qmin', 'qmax', 'close', 'stake'])
    ref = df[df['pool']]
    parts = {c: np.sort(ref[c].to_numpy()) for c in ('qmin', 'qmax', 'close')}
    q_raw = sum(w * _pct(parts[c], df[c].to_numpy()) for c, w in (('qmin', .35), ('qmax', .35), ('close', .3)))
    q = _pct(np.sort(q_raw[df['pool'].to_numpy()]), q_raw) * 100
    st_ref = np.sort(ref['stake'].dropna().to_numpy())
    for (g, *_), qq, ss in zip(rows, q, df['stake']):
        g['quality'] = int(round(min(qq, 100)))
        if pd.isna(ss):
            g['stakes_score'] = g['juice'] = None
        else:
            s_pct = min(_pct(st_ref, ss) * 100, 100)
            g['stakes_score'] = int(round(s_pct))
            g['juice'] = int(round(np.sqrt(min(qq, 100) * s_pct)))


# ── Per-season cache ─────────────────────────────────────────────────────────
# Finished seasons never change unless the engine or their inputs do: cache
# each season under a fingerprint of this file + the sim's engine files + the
# season's games (and last season's, for opening night and the scoring
# baseline) + ratings (rounded to 3dp; the ratings engine isn't
# bit-reproducible) + Cup finals + start times.
CACHE_DIR = 'matchups_cache'
_HERE = os.path.dirname(os.path.abspath(__file__))
_JOB = {}


def _fingerprint(season, J):
    h = hashlib.sha256()
    for f in ('daily_matchups.py',) + playoff_sim._ENGINE_FILES:
        h.update(open(os.path.join(_HERE, f), 'rb').read())
    g = J['sim_games'][J['sim_games']['season'].isin([season - 1, season])]
    h.update(g.sort_values(list(g.columns), kind='stable', na_position='last').to_csv(index=False).encode())
    r = J['ratings'][J['ratings']['season'].isin([season - 1, season])].sort_values(['date', 'name']).copy()
    for c in ('rating', 'rating_o', 'rating_d'):
        r[c] = r[c].round(3)
    h.update(r.to_csv(index=False).encode())
    h.update(J['cup'][J['cup']['season'] == season].to_csv(index=False).encode())
    h.update(repr(sorted((k, v) for k, v in J['start_times'].items() if k[0].year in (season - 1, season))).encode())
    h.update(repr((J['rs_end'].get(season), J['rs_games'](season), J['prev_mu'].get(season))).encode())
    return h.hexdigest()


def _one(season):
    J = _JOB
    r = J['ratings']
    days = build_season(season, J['sim_games'], r[r['season'].isin([season - 1, season])],
                        J['rs_games'](season), J['conf_of'], J['div_of'], J['rs_end'].get(season),
                        J['cup'][J['cup']['season'] == season], J['start_times'], J['prev_mu'][season],
                        log=lambda *_: None)
    return season, days


def build_cached(seasons, sim_games, ratings, rs_games, conf_of, div_of, rs_end, cup, start_times,
                 workers=None, log=print):
    """{season: days} for every season, reusing cached ones. ratings needs
    season, date, name, rating, rank, rating_o, rating_d."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    played = sim_games[sim_games['home_pts'].notna()]
    team_pts = pd.concat([played[['season', 'home_pts']].rename(columns={'home_pts': 'p'}),
                          played[['season', 'visitor_pts']].rename(columns={'visitor_pts': 'p'})])
    season_mu = team_pts.groupby('season')['p'].mean().to_dict()
    first = min(season_mu)
    prev_mu = {s: float(season_mu.get(s - 1, season_mu.get(s, season_mu[first]))) for s in seasons}
    _JOB.update(sim_games=sim_games, ratings=ratings, rs_games=rs_games, conf_of=conf_of, div_of=div_of,
                rs_end=rs_end, cup=cup, start_times=start_times, prev_mu=prev_mu)
    out, todo, sigs = {}, [], {}
    for s in seasons:
        sigs[s] = _fingerprint(s, _JOB)
        path = os.path.join(CACHE_DIR, f'{s}.pkl')
        if os.path.exists(path):
            try:
                old_sig, days = pickle.load(open(path, 'rb'))
                if old_sig == sigs[s]:
                    out[s] = days
                    continue
            except Exception:
                pass
        todo.append(s)
    log(f'  daily matchups: {len(seasons) - len(todo)} seasons from cache, computing {len(todo)}')
    if todo:
        with _mp.get_context('fork').Pool(workers or os.cpu_count()) as pool:
            for s, days in pool.imap_unordered(_one, todo):
                out[s] = days
                pickle.dump((sigs[s], days), open(os.path.join(CACHE_DIR, f'{s}.pkl'), 'wb'))
    return {s: out[s] for s in seasons}
