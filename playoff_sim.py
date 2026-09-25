"""DUNCAN title odds: Monte Carlo of the rest of the NBA season + playoffs.

Port of LOBO's playoff_sim.py. For every rating snapshot, simulate the
remaining regular-season games, seed that season's playoff format from the
simulated standings (division-winner rules included), then play the
bracket, play-in included. Anything already played is fixed.

Game model (probit on 58,577 NBA games 1977-2026, pre-game snapshot
ratings, fit by log loss per era):
    P(home win) = Phi(A * (rating_home - rating_away + home_pts))
Both A and home court drifted, so they're per era (see ERA_PARAMS). The
2020 bubble (games from 2020-07-30) is neutral.

Output per (snapshot, team): probability of reaching the playoff bracket,
each later round, and the title (the Title odds column).
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr

# Simulation counts (fleet standard): regular-season dates 10k; once the
# regular season is over, 100k (10k left a visible ~1-point day-to-day
# wobble in the playoff grid; 100k is ~0.1 pt).
N_SIMS = 10_000
N_SIMS_PLAYOFFS = 100_000
# Large runs go in chunks (like MESSI's 200k): a single 1M-sim call builds
# several multi-GB arrays for a 30-team league, and the first full build
# lost its 2026 worker running ten of those in parallel.
CHUNK = 200_000

# (first season, A, home court pts), fit per decade.
ERA_PARAMS = [
    (1977, 0.0889, 4.98),
    (1990, 0.0913, 3.89),
    (2000, 0.0843, 3.67),
    (2010, 0.0829, 3.17),
    (2021, 0.0726, 2.05),
]
BUBBLE_START = pd.Timestamp('2020-07-30')

# Standings ties the NBA broke differently from our rule-based tiebreak
# (older eras used other rules, including coin flips; multi-team ties are
# intricate). Per season, tied teams in the order they were actually
# seeded; earlier wins. Recovered by searching tie orderings for the one
# whose bracket reproduces every real playoff game (search_tiebreaks.py).
import json as _json
import os as _os
_TB = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'nba_tiebreak_orders.json')
TIEBREAK_WINNERS = ({int(k): v for k, v in _json.load(open(_TB)).items()}
                    if _os.path.exists(_TB) else {})


def era_params(season):
    a, h = ERA_PARAMS[0][1:]
    for start, aa, hh in ERA_PARAMS:
        if season >= start:
            a, h = aa, hh
    return a, h


def division_rule(season):
    """How division winners are seeded within a conference."""
    if season <= 2004:
        return 'top2'   # division winners seeded 1-2
    if season <= 2006:
        return 'top3'   # three division winners seeded 1-3
    if season <= 2015:
        return 'top4'   # division winners guaranteed a top-4 seed
    return None


# ── Formats ──────────────────────────────────────────────────────────────────
# Slots: ('conf', c, k) = k-th seed in conference c; ('W', id) / ('L', id) =
# that match's winner / loser. Matches: (id, round, kind, slot_a, slot_b)
# where kind is a best-of length or 'pi2020' (2020 play-in: the 8 seed
# needs one win, the 9 seed two, and it only happens if the 9 seed is
# within four games).
def _conf(c, first_bo, twelve=False, playin=None):
    k = c[0]
    m = []
    if twelve:  # 1977-1983: 6 per conference, top two seeds bye
        m += [(f'{k}a', 1, 3, ('conf', c, 3), ('conf', c, 6)),
              (f'{k}b', 1, 3, ('conf', c, 4), ('conf', c, 5)),
              (f'{k}S1', 2, 7, ('conf', c, 1), ('W', f'{k}b')),
              (f'{k}S2', 2, 7, ('conf', c, 2), ('W', f'{k}a')),
              (f'{k}F', 3, 7, ('W', f'{k}S1'), ('W', f'{k}S2'))]
        return m
    r0 = 0
    eight = ('conf', c, 8)
    seven = ('conf', c, 7)
    if playin == 'tournament':  # 2021+
        r0 = 1
        m += [(f'{k}PA', 1, 1, ('conf', c, 7), ('conf', c, 8)),
              (f'{k}PB', 1, 1, ('conf', c, 9), ('conf', c, 10)),
              (f'{k}PC', 1, 1, ('L', f'{k}PA'), ('W', f'{k}PB'))]
        seven, eight = ('W', f'{k}PA'), ('W', f'{k}PC')
    elif playin == '2020':
        r0 = 1
        m += [(f'{k}PA', 1, 'pi2020', ('conf', c, 8), ('conf', c, 9))]
        eight = ('W', f'{k}PA')
    m += [(f'{k}A', r0 + 1, first_bo, ('conf', c, 1), eight),
          (f'{k}B', r0 + 1, first_bo, ('conf', c, 4), ('conf', c, 5)),
          (f'{k}C', r0 + 1, first_bo, ('conf', c, 3), ('conf', c, 6)),
          (f'{k}D', r0 + 1, first_bo, ('conf', c, 2), seven),
          (f'{k}S1', r0 + 2, 7, ('W', f'{k}A'), ('W', f'{k}B')),
          (f'{k}S2', r0 + 2, 7, ('W', f'{k}D'), ('W', f'{k}C')),
          (f'{k}F', r0 + 3, 7, ('W', f'{k}S1'), ('W', f'{k}S2'))]
    return m


def bracket_for(season):
    if season <= 1983:
        b = _conf('East', 3, twelve=True) + _conf('West', 3, twelve=True)
    else:
        first = 5 if season <= 2002 else 7
        pi = '2020' if season == 2020 else ('tournament' if season >= 2021 else None)
        b = _conf('East', first, playin=pi) + _conf('West', first, playin=pi)
    last = max(r for _, r, *_ in b) + 1
    return b + [('F', last, 7, ('W', 'EF'), ('W', 'WF'))]


def host_pattern(season, best_of, finals=False):
    """Per game: True = better record hosts."""
    if best_of == 1:
        return [True]
    if best_of == 3:
        return [True, False, True]
    if best_of == 5:
        return [True, True, False, False, True]
    if finals and 1985 <= season <= 2013:
        return [True, True, False, False, False, True, True]   # 2-3-2
    return [True, True, False, False, True, False, True]


def round_names(season):
    """(full, short) names for every round, first to last."""
    if season >= 2020:
        return (['Play-In', 'First Round', 'Conference Semifinals', 'Conference Finals', 'Finals'],
                ['Play-In', 'R1', 'R2', 'Conf Final', 'Final'])
    return (['First Round', 'Conference Semifinals', 'Conference Finals', 'Finals'],
            ['R1', 'R2', 'Conf Final', 'Final'])


def entry_rounds(season):
    """Seed slot label -> round that seed enters (later than 1 = bye)."""
    out = {}
    for _, rnd, _, *slots in bracket_for(season):
        for sl in slots:
            if sl[0] == 'conf':
                out.setdefault(f"{sl[1][0]}{sl[2]}", rnd)
    return out


class SeasonSim:
    def __init__(self, season, games, rs_games_total, conf_of, div_of, ratings, rs_end=None):
        """games: season's games (date, home, away, home_pts, visitor_pts; NaN
        pts = scheduled). Regular season = games on or before rs_end when
        known (completed seasons), else each team's first rs_games_total."""
        self.season = season
        g = games.sort_values('date', kind='stable').reset_index(drop=True)
        if rs_end is not None:
            g['is_rs'] = g['date'] <= rs_end
        else:
            counts, is_rs = {}, []
            for h, a in zip(g['home'], g['away']):
                ch, ca = counts.get(h, 0) + 1, counts.get(a, 0) + 1
                counts[h], counts[a] = ch, ca
                is_rs.append(ch <= rs_games_total and ca <= rs_games_total)
            g['is_rs'] = is_rs
        rs = g[g['is_rs']]
        self.teams = sorted(set(rs['home']) | set(rs['away']))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.conf = np.array([conf_of(t, season) for t in self.teams])
        self.div = np.array([div_of(t, season) for t in self.teams])
        self.A, self.hp = era_params(season)
        rs = rs.assign(h=rs['home'].map(self.idx), a=rs['away'].map(self.idx))
        rs['hp'] = np.where((season == 2020) & (rs['date'] >= BUBBLE_START), 0.0, self.hp)
        self.rs = rs
        self.hp_ps = 0.0 if season == 2020 else self.hp
        self.ps = g[~g['is_rs'] & g['home_pts'].notna()].copy()
        self.ps['winner'] = np.where(self.ps['home_pts'] > self.ps['visitor_pts'],
                                     self.ps['home'], self.ps['away'])
        self.ratings = ratings
        self.bracket = bracket_for(season)
        self.n_rounds = max(r for _, r, *_ in self.bracket)

    def rs_over(self, d):
        return not ((self.rs['date'] > d) | self.rs['home_pts'].isna()).any()

    def _static_tiebreak(self, done):
        """Deterministic tiebreak once the regular season is complete, NBA
        order within each group tied on win%: two teams = head-to-head, then
        division leader, then conference record; three or more = division
        leader first, then head-to-head within the group, then conference
        record. Point differential last. Returns a per-team score (higher =
        better); only compared within a tie group."""
        T = len(self.teams)
        w = np.zeros(T); gp = np.zeros(T); pd_ = np.zeros(T); cw = np.zeros(T); cg = np.zeros(T)
        for h, a, hp, vp in done[['h', 'a', 'home_pts', 'visitor_pts']].itertuples(index=False):
            win = h if hp > vp else a
            gp[h] += 1; gp[a] += 1; pd_[h] += hp - vp; pd_[a] += vp - hp; w[win] += 1
            if self.conf[h] == self.conf[a]:
                cg[h] += 1; cg[a] += 1; cw[win] += 1
        pct = w / np.maximum(gp, 1)
        cpct = cw / np.maximum(cg, 1)
        leader = np.zeros(T)
        for c in np.unique(self.conf):
            for dv in np.unique(self.div[self.conf == c]):
                mem = np.where((self.conf == c) & (self.div == dv))[0]
                if dv == 'none' or not len(mem):
                    continue
                leader[mem[np.argmax(pct[mem] * 1e6 + pd_[mem])]] = 1
        score = np.zeros(T)
        for p in np.unique(pct):
            grp = np.where(pct == p)[0]
            if len(grp) < 2:
                continue
            gs = set(grp)
            sub = done[done['h'].isin(gs) & done['a'].isin(gs)]
            hw = np.zeros(T); hg = np.zeros(T)
            for h, a, hp, vp in sub[['h', 'a', 'home_pts', 'visitor_pts']].itertuples(index=False):
                hg[h] += 1; hg[a] += 1; hw[h if hp > vp else a] += 1
            h2h = np.where(hg > 0, hw / np.maximum(hg, 1), 0.5)
            # Head-to-head then division leader for two teams matched more real
            # brackets (36 of 50 seasons) than division leader first (32).
            first, second = (h2h, leader) if len(grp) == 2 else (leader, h2h)
            key = sorted(grp, key=lambda t: (first[t], second[t], cpct[t], pd_[t]), reverse=True)
            for rank, t in enumerate(key):
                score[t] = len(key) - rank
        # Recorded outcomes win over the rules (earlier in the list = higher).
        order = TIEBREAK_WINNERS.get(self.season, [])
        for k, t in enumerate(order):
            score[self.idx[t]] += 1e6 * (len(order) - k)
        return score

    def odds_at(self, d, n_sims=N_SIMS):
        """Chunked wrapper: same result shape, bounded memory."""
        if n_sims <= CHUNK:
            return self._odds_at(d, n_sims, 0)
        parts, done = [], 0
        k = 0
        while done < n_sims:
            n = min(CHUNK, n_sims - done)
            parts.append(self._odds_at(d, n, k) * n)
            done += n; k += 1
        return sum(parts) / n_sims

    def _odds_at(self, d, n_sims, chunk):
        T = len(self.teams)
        rng = np.random.default_rng([int(pd.Timestamp(d).strftime('%Y%m%d')), chunk])
        rt = self.ratings.get(d, {})
        R = np.array([rt.get(t, 0.0) for t in self.teams])
        A = self.A

        played = self.rs['home_pts'].notna() & (self.rs['date'] <= d)
        done, rest = self.rs[played], self.rs[~played]
        w0 = np.zeros(T); g0 = np.zeros(T)
        for h, a, hpt, vpt in done[['h', 'a', 'home_pts', 'visitor_pts']].itertuples(index=False):
            g0[h] += 1; g0[a] += 1; w0[h if hpt > vpt else a] += 1
        W = np.tile(w0, (n_sims, 1)); G = np.tile(g0, (n_sims, 1))
        if len(rest):
            h = rest['h'].to_numpy(); a = rest['a'].to_numpy()
            ph = ndtr(A * (R[h] - R[a] + rest['hp'].to_numpy()))
            hw = (rng.random((n_sims, len(rest))) < ph).astype(np.float32)
            Hm = np.zeros((len(rest), T), np.float32); Hm[np.arange(len(rest)), h] = 1
            Am = np.zeros((len(rest), T), np.float32); Am[np.arange(len(rest)), a] = 1
            W += hw @ Hm + (1 - hw) @ Am
            G += (Hm + Am).sum(0)
        pct = W / np.maximum(G, 1)
        static = self._static_tiebreak(done) if rest.empty else np.zeros(T)
        tieb = np.tile(static, (n_sims, 1)) + rng.random((n_sims, T)) * 1e-6
        sim_ix = np.arange(n_sims)

        def ranked(members, bonus=None):
            m = np.array(members)
            k = len(m)
            keys = [(-tieb[:, m]).ravel(), (-pct[:, m]).ravel()]
            if bonus is not None:
                keys.append((-bonus).ravel())
            order = np.lexsort(keys + [np.repeat(sim_ix, k)])
            return m[order.reshape(n_sims, k) % k]

        rule = division_rule(self.season)
        by_conf = {}
        for c in ('East', 'West'):
            m = np.where(self.conf == c)[0]
            order = ranked(m)
            if rule:
                # Division winner = best-placed division member by record.
                pos = np.empty((n_sims, T), dtype=int)
                pos[sim_ix[:, None], order] = np.arange(len(m))[None, :]
                bonus = np.zeros((n_sims, len(m)))
                col = np.full(T, -1); col[m] = np.arange(len(m))
                for dv in np.unique(self.div[m]):
                    mem = m[self.div[m] == dv]
                    win = mem[np.argmin(pos[:, mem], axis=1)]
                    bonus[sim_ix, col[win]] = 1
                if rule == 'top4':
                    # Plus the best non-division-winner.
                    nonw = np.where(bonus == 0, pos[:, m], 10**6)
                    bonus[sim_ix, np.argmin(nonw, axis=1)] = 1
                order = ranked(m, bonus)
            by_conf[c] = order
        lg_all = ranked(range(T))
        lg_rank = np.empty((n_sims, T), dtype=int)
        lg_rank[sim_ix[:, None], lg_all] = np.arange(T)[None, :]

        ps_by_pair = {}
        for r in self.ps[self.ps['date'] <= d].itertuples(index=False):
            ps_by_pair.setdefault(frozenset((r.home, r.away)), []).append(r.winner)

        self.used_actual = 0
        self.rs_complete = rest.empty
        self.seeds = {}
        if self.rs_complete:
            for c, arr in by_conf.items():
                for k, t in enumerate(arr[0]):
                    self.seeds[self.teams[t]] = f"{c[0]}{k + 1}"
        self.matchups = []  # (round, kind, team_a, team_b, winners so far, decided winner)

        reach = np.zeros((self.n_rounds + 2, T))
        entered = np.zeros((n_sims, T), dtype=bool)
        res, lose = {}, {}

        def slot(s):
            if s[0] == 'conf':
                return by_conf[s[1]][:, s[2] - 1]
            return res[s[1]] if s[0] == 'W' else lose[s[1]]

        for mid, rnd, kind, sa, sb in self.bracket:
            a, b = slot(sa), slot(sb)
            for t in (a, b):
                new = ~entered[sim_ix, t]
                entered[sim_ix, t] = True
                np.add.at(reach[0], t[new], 1)
                for k in range(2, rnd):   # a bye counts as getting through
                    np.add.at(reach[k], t[new], 1)
                np.add.at(reach[rnd], t, 1)
            a_better = lg_rank[sim_ix, a] < lg_rank[sim_ix, b]
            fixed = np.all(a == a[0]) and np.all(b == b[0])
            actual = ps_by_pair.get(frozenset((self.teams[a[0]], self.teams[b[0]])), []) if fixed else []
            p_a = lambda edge: ndtr(A * (R[a] - R[b] + edge))
            if kind == 'pi2020':
                # a = 8 seed (one win needed), b = 9 seed (two wins needed);
                # skipped entirely if the 9 seed finished more than 4 games back.
                gb = ((W[sim_ix, a] - W[sim_ix, b]) + ((G - W)[sim_ix, b] - (G - W)[sim_ix, a])) / 2
                held = gb <= 4
                wa = np.zeros(n_sims, dtype=int); wb = np.zeros(n_sims, dtype=int)
                for gi in range(2):
                    if gi < len(actual):
                        won = np.full(n_sims, actual[gi] == self.teams[a[0]]); self.used_actual += 1
                    else:
                        won = rng.random(n_sims) < p_a(self.hp_ps)
                    live = held & (wa < 1) & (wb < 2)
                    wa += won & live; wb += ~won & live
                a_wins = ~held | (wa >= 1)
            else:
                bo = kind
                need = bo // 2 + 1
                wa = np.zeros(n_sims, dtype=int); wb = np.zeros(n_sims, dtype=int)
                for gi, better_hosts in enumerate(host_pattern(self.season, bo, finals=(mid == 'F'))):
                    if gi < len(actual):
                        won = np.full(n_sims, actual[gi] == self.teams[a[0]]); self.used_actual += 1
                    else:
                        a_home = a_better if better_hosts else ~a_better
                        won = rng.random(n_sims) < p_a(np.where(a_home, self.hp_ps, -self.hp_ps))
                    live = (wa < need) & (wb < need)
                    wa += won & live; wb += ~won & live
                a_wins = wa >= need
            res[mid] = np.where(a_wins, a, b)
            lose[mid] = np.where(a_wins, b, a)
            if fixed and self.rs_complete and not (kind == 'pi2020' and not held[0]):
                ta, tb = self.teams[a[0]], self.teams[b[0]]
                games_ = list(actual[:2 if kind == 'pi2020' else kind])
                na, nb = games_.count(ta), games_.count(tb)
                if kind == 'pi2020':
                    decided = ta if na >= 1 else (tb if nb >= 2 else None)
                else:
                    decided = ta if na >= kind // 2 + 1 else (tb if nb >= kind // 2 + 1 else None)
                self.matchups.append((rnd, 2 if kind == 'pi2020' else kind, ta, tb, games_, decided))
        np.add.at(reach[-1], res['F'], 1)
        reach /= n_sims
        cols = ['playoffs'] + [f'r{k}' for k in range(2, self.n_rounds + 1)] + ['champ']
        rows = np.vstack([reach[0]] + [reach[k] for k in range(2, self.n_rounds + 1)] + [reach[-1]])
        return pd.DataFrame(rows.T, index=self.teams, columns=cols)


def compute(games, ratings_df, rs_games_by_season, conf_of, div_of, current_season,
            rs_end_by_season, seasons=None, log=print):
    """games: all NBA games (season, date, home, away, home_pts, visitor_pts),
    NBA Cup final excluded, scheduled games with NaN points.
    ratings_df: (season, date, name, rating). Returns (odds, brackets):
    odds = long DataFrame (season, date, team, playoffs, r2.., champ);
    brackets = {season: {date: (seeds, matchups, n_sims)}} for snapshots
    on or after the end of the regular season."""
    out = []
    brackets = {}
    for season, g in games.groupby('season'):
        season = int(season)
        if seasons is not None and season not in seasons:
            continue
        rsub = ratings_df[ratings_df['season'] == season]
        ratings = {d: dict(zip(x['name'], x['rating'])) for d, x in rsub.groupby('date')}
        if not ratings:
            continue
        sim = SeasonSim(season, g, rs_games_by_season(season), conf_of, div_of, ratings,
                        rs_end_by_season.get(season))
        for d in sorted(ratings):
            n = N_SIMS
            if sim.rs_over(d):
                n = N_SIMS_PLAYOFFS
            o = sim.odds_at(d, n_sims=n)
            if sim.rs_complete:
                brackets.setdefault(season, {})[d] = (dict(sim.seeds), list(sim.matchups), n)
            o.index.name = 'team'
            o = o.reset_index()
            o['season'] = season
            o['date'] = d
            out.append(o)
        log(f"  {season}: {len(ratings)} snapshots")
    return pd.concat(out, ignore_index=True), brackets


# ── Cached, parallel driver ──────────────────────────────────────────────────
# A full NBA history takes ~80 min single-threaded (50 seasons x ~200
# snapshots), but finished seasons never change unless the engine or their
# inputs do. Each season's result is cached under a fingerprint of the
# engine code + tiebreak/division files + that season's games and ratings
# (ratings rounded to 3dp: the ratings engine isn't bit-reproducible), so
# any change to those recomputes that season - no stale-history trap.
import hashlib
import multiprocessing as _mp
import pickle

_ENGINE_FILES = ('playoff_sim.py', 'nba_tiebreak_orders.json', 'nba_divisions.csv')
_JOB = {}


def _fingerprint(season, games, ratings_df, current_season):
    h = hashlib.sha256()
    here = _os.path.dirname(_os.path.abspath(__file__))
    for f in _ENGINE_FILES:
        p = _os.path.join(here, f)
        if _os.path.exists(p):
            h.update(open(p, 'rb').read())
    g = games[games['season'] == season].sort_values(['date', 'home']).copy()
    h.update(g.to_csv(index=False).encode())
    r = ratings_df[ratings_df['season'] == season].sort_values(['date', 'name']).copy()
    r['rating'] = r['rating'].round(3)
    h.update(r.to_csv(index=False).encode())
    return h.hexdigest()


def _one(season):
    j = _JOB
    return season, compute(j['games'], j['ratings'], j['rsg'], j['conf_of'], j['div_of'],
                           j['current'], j['rs_end'], seasons={season}, log=lambda *_: None)


def compute_cached(games, ratings_df, rs_games_by_season, conf_of, div_of, current_season,
                   rs_end_by_season, cache_dir='title_odds_cache', workers=None, log=print):
    """compute() over every season, reusing cached seasons whose fingerprint
    still matches and recomputing the rest in parallel."""
    _os.makedirs(cache_dir, exist_ok=True)
    seasons = sorted(int(s) for s in games['season'].unique()
                     if (ratings_df['season'] == s).any())
    results, todo, sigs = {}, [], {}
    for s in seasons:
        sigs[s] = _fingerprint(s, games, ratings_df, current_season)
        path = _os.path.join(cache_dir, f'{s}.pkl')
        if _os.path.exists(path):
            try:
                sig, payload = pickle.load(open(path, 'rb'))
                if sig == sigs[s]:
                    results[s] = payload
                    continue
            except Exception:
                pass
        todo.append(s)
    log(f"  {len(results)} seasons from cache, computing {len(todo)}: {todo}")
    if todo:
        _JOB.update(games=games, ratings=ratings_df, rsg=rs_games_by_season, conf_of=conf_of,
                    div_of=div_of, current=current_season, rs_end=rs_end_by_season)
        ctx = _mp.get_context('fork')   # workers inherit _JOB; no re-import of the caller
        with ctx.Pool(workers or _os.cpu_count()) as pool:
            for s, payload in pool.imap_unordered(_one, todo):
                results[s] = payload
                pickle.dump((sigs[s], payload), open(_os.path.join(cache_dir, f'{s}.pkl'), 'wb'))
                log(f"  {s} done")
    odds = pd.concat([results[s][0] for s in seasons], ignore_index=True)
    brackets = {}
    for s in seasons:
        brackets.update(results[s][1])
    return odds, brackets
