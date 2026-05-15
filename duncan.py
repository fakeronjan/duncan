# =========================================================
# DUNCAN NBA POWER RATINGS
# =========================================================

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date

# =========================================================
# CONFIGURATION
# =========================================================

MIN_SEASON = 1980             # DO NOT CHANGE — affects all historical data

# Season-aware rolling window: window (game-days) = WINDOW_MULTIPLIER * games-per-team-per-season.
# At 1.5, an 82-game NBA season gets a 123-day window (was fixed 100 pre-port).
# Lockout/COVID seasons get proportionally smaller windows automatically.
WINDOW_MULTIPLIER = 1.5

HOME_COURT_ADJUSTMENT = 2.0   # raw-point home advantage, subtracted from home margin pre-transform

# Margin transform: cap at p~92 of NBA margins. Linear (raw-points-as-rating)
# for the bulk of games; clipped above to keep blowouts from dominating.
MARGIN_TRANSFORM = "cap"
MARGIN_CAP = 25

# WLS: weights affect observation influence, not margin magnitude.
WEIGHTING_MODE = "wls"

# Re-process the most recent N ranking_ids (game-days) on every run so late-
# arriving NBA data is absorbed. Without this a mid-day cron caches that
# day's snapshot and never re-ranks it even when more games for the same
# day finish hours later.
RECOMPUTE_TAIL_DAYS = 7

# Regular season game count per season (drives both window sizing and playoff-start
# threshold). NBA is normally 82 games; lockouts/COVID exceptions noted inline.
REGULAR_SEASON_GAMES = {
    **{y: 82 for y in range(1980, 1999)},
    1999: 50,  # lockout
    **{y: 82 for y in range(2000, 2012)},
    2012: 66,  # lockout
    **{y: 82 for y in range(2013, 2020)},
    2020: 72,  # COVID
    2021: 72,  # post-COVID
    **{y: 82 for y in range(2022, 2030)},
}

# Emirates NBA Cup championship games. The NBA does NOT count this game in
# the regular-season W-L record for either participant — both finalists
# play 82 RS games on top of this one. The data still includes the game
# (it's a real on-court signal for ratings) but standings exclude it.
# Season uses the calendar year of the Finals (2024 = the 2023-24 season).
NBA_CUP_FINAL_DATES = {
    '2023-12-09',  # 2023-24 NBA Cup
    '2024-12-17',  # 2024-25 NBA Cup
    '2025-12-16',  # 2025-26 NBA Cup
}

# Same-market rebrand consolidation. Maps historical team names to current
# canonical names so a single franchise's history reads as one team across
# rebrands. RELOCATIONS are deliberately kept separate ("move the team =
# lose the history" policy): Baltimore Colts → Indianapolis, Kansas City
# Kings → Sacramento, Seattle SuperSonics → OKC, San Diego Clippers → LA,
# Vancouver Grizzlies → Memphis, NJ Nets → Brooklyn, etc. all remain
# distinct from their post-move franchises.
#
# Charlotte note: the original Charlotte Hornets (1989-2002) relocated to
# New Orleans in 2002. The modern Charlotte Hornets (2014+) are the
# rebranded Bobcats franchise, which the NBA officially credits with the
# 1989-2002 Hornets history. Our source data already merges those two
# eras under "Charlotte Hornets" (same LA-Rams-style quirk we accepted
# earlier) — we just consolidate Bobcats on top of that.
TEAM_ALIASES = {
    'Washington Bullets':                'Washington Wizards',
    'Charlotte Bobcats':                  'Charlotte Hornets',
    'New Orleans Hornets':                'New Orleans Pelicans',
    'New Orleans/Oklahoma City Hornets':  'New Orleans Pelicans',
}

# =========================================================
# SCRAPING
# =========================================================

MONTHS = [
    'october', 'november', 'december', 'january', 'february',
    'march', 'april', 'may', 'june', 'july', 'august', 'september'
]


def scrape_table(url, year):
    """Scrape a single basketball-reference schedule page into a DataFrame."""
    r = requests.get(url)
    if '404' in str(r):
        return pd.DataFrame()

    html = BeautifulSoup(r.text, "lxml")
    columns = [i['data-stat'] for i in html.select('#schedule > thead > tr > th')]
    data = {k: [] for k in columns}

    for row in html.select('#schedule > tbody > tr'):
        for entry in row:
            if 'Playoffs' in entry:
                break
            data[entry['data-stat']].append(entry.get_text())

    df = pd.DataFrame(data)
    df['season'] = year
    return df


def scrape_games(min_season, max_season, existing_df):
    """
    Scrape any seasons not already fully captured in existing_df.
    Returns a combined DataFrame of all games (old + new), saved to loaded_nba_games.csv.
    """
    max_season_completed = max(existing_df['season']) - 1  # latest season may be partial
    min_season_completed = min(existing_df['season'])

    print(f"Already have complete data for seasons {min_season_completed}–{max_season_completed}")
    print(f"Checking for new data through season {max_season}")

    new_frames = []
    for year in range(min_season, max_season + 1):
        if min_season_completed <= year <= max_season_completed:
            continue
        for month in MONTHS:
            url = f'https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html'
            df = scrape_table(url, year)
            new_frames.append(df)
        print(f"{year} — scraped!")

    combined = pd.concat([existing_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    combined.sort_values('season', inplace=True)
    combined.drop_duplicates(keep="first", inplace=True)
    combined.to_csv('loaded_nba_games.csv', index=False)
    return combined


# =========================================================
# GAME DATA PREPARATION
# =========================================================

def prepare_game_data(raw_df):
    """
    Clean and enrich the raw games DataFrame with margins, win flags,
    adjusted scores, date IDs, and result strings.
    """
    df = raw_df[['season', 'date_game', 'visitor_team_name', 'visitor_pts', 'home_team_name', 'home_pts']].copy()

    df['visitor_pts'] = pd.to_numeric(df['visitor_pts'])
    df['home_pts'] = pd.to_numeric(df['home_pts'])

    # Apply same-market rebrand consolidation before any team-keyed work
    # (margins, result strings, downstream merges). Old names show up
    # rendered as the franchise's current name everywhere internally;
    # generate_data.py re-applies an era-appropriate display label per row.
    df['visitor_team_name'] = df['visitor_team_name'].replace(TEAM_ALIASES)
    df['home_team_name']    = df['home_team_name'].replace(TEAM_ALIASES)

    # Margin of victory (raw points). HCA and the margin transform are
    # applied inside the solver, not here, so downstream consumers see
    # the unmodified game record.
    df['visitor_margin'] = df['visitor_pts'] - df['home_pts']
    df['home_margin'] = -df['visitor_margin']

    # Win flags
    df['visitor_win'] = np.where(df['visitor_margin'] > 0, 1, 0)
    df['home_win'] = 1 - df['visitor_win']

    # Drop incomplete rows before date parsing
    df = df.dropna()

    # Date parsing and sorting
    df['date_game'] = pd.to_datetime(df['date_game'], format='%a, %b %d, %Y')
    df.sort_values('date_game', inplace=True)
    df.drop_duplicates(keep="first", inplace=True)

    # Date and game IDs
    df['grouped_date_id'] = df.groupby(['date_game']).ngroup() + 1
    df['unique_game_id'] = df.groupby(df.columns.tolist(), sort=False).ngroup() + 1

    # Result strings
    df['home_pts'] = df['home_pts'].astype(int)
    df['visitor_pts'] = df['visitor_pts'].astype(int)
    df['home_wl'] = np.where(df['home_win'] == 1, "W", "L")
    df['visitor_wl'] = np.where(df['visitor_win'] == 1, "W", "L")
    df['home_result'] = (
        df['home_wl'] + " vs. " + df['visitor_team_name'] + " "
        + df['home_pts'].map(str) + "-" + df['visitor_pts'].map(str)
    )
    df['visitor_result'] = (
        df['visitor_wl'] + " @ " + df['home_team_name'] + " "
        + df['visitor_pts'].map(str) + "-" + df['home_pts'].map(str)
    )

    # Flag NBA Cup championship games (excluded from regular-season W-L counts).
    df['is_nba_cup_final'] = df['date_game'].astype(str).isin(NBA_CUP_FINAL_DATES).astype(int)

    df.to_csv('all_nba_games.csv', index=False)
    print("CSV of NBA games is ready!")
    return df


# =========================================================
# MASSEY RATINGS — homebrew weighted least squares solver
# =========================================================

def _apply_margin_transform(margin, transform, cap):
    """Sign-preserving transform applied to (raw_margin - hca)."""
    m = np.asarray(margin, dtype=float)
    if transform == "raw":
        return m
    if transform == "sqrt":
        return np.sign(m) * np.sqrt(np.abs(m))
    if transform == "cap":
        return np.clip(m, -cap, cap)
    if transform == "log":
        return np.sign(m) * np.log1p(np.abs(m))
    if transform == "tanh":
        return cap * np.tanh(m / cap)
    raise ValueError(f"Unknown MARGIN_TRANSFORM: {transform}")


def _solve_massey(window_df, hca, weighting_mode, margin_transform, margin_cap):
    """
    Solve for team Massey ratings on a single rolling window.

    Builds X (n_games × n_teams) with +1 for home, -1 for visitor, y from
    the transformed HCA-adjusted home margin, and W from the recency
    weights. Solves min sum_i w_i * (X_i r - y_i)^2 with a zero-sum
    constraint enforced as an extra high-weight row.

    WLS via row-scaling: multiplying both X and y by sqrt(w_i) turns the
    weighted problem into an ordinary lstsq.
    """
    teams = sorted(set(window_df["home_team_name"]) | set(window_df["visitor_team_name"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    X = np.zeros((n_games + 1, n_teams))
    y = np.zeros(n_games + 1)
    w = np.zeros(n_games + 1)

    home_pts = window_df["home_pts"].to_numpy(dtype=float)
    visitor_pts = window_df["visitor_pts"].to_numpy(dtype=float)
    weights = window_df["date_weight"].to_numpy(dtype=float)
    home_names = window_df["home_team_name"].to_numpy()
    visitor_names = window_df["visitor_team_name"].to_numpy()

    raw_margin = home_pts - visitor_pts - hca
    transformed = _apply_margin_transform(raw_margin, margin_transform, margin_cap)

    for i in range(n_games):
        X[i, team_idx[home_names[i]]] = 1.0
        X[i, team_idx[visitor_names[i]]] = -1.0

    if weighting_mode == "wls":
        y[:n_games] = transformed
        w[:n_games] = weights
    elif weighting_mode == "margin_scale":
        y[:n_games] = transformed * weights
        w[:n_games] = 1.0
    else:
        raise ValueError(f"Unknown WEIGHTING_MODE: {weighting_mode}")

    # Zero-sum constraint via high-weight extra row.
    X[-1, :] = 1.0
    y[-1] = 0.0
    w[-1] = 1.0e8

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = pd.DataFrame({"name": teams, "rating": r})
    out["rank"] = out["rating"].rank(ascending=False, method="min").astype(int)
    return out


def _window_for_season(season):
    """Season-aware window size: WINDOW_MULTIPLIER × regular-season games per team."""
    reg_games = REGULAR_SEASON_GAMES.get(int(season), 82)
    return int(round(reg_games * WINDOW_MULTIPLIER))


# Smallest window across any season - floor for the loop's starting ranking_id.
# Using the max here would silently drop any season whose total game-days are
# shorter than the modern window (would bite the 1999 / 2012 lockouts if
# WINDOW_MULTIPLIER ever moved up). Each ranking_id is then gated by its OWN
# season's window_size inside the loop.
_MIN_WINDOW = min(_window_for_season(s) for s in REGULAR_SEASON_GAMES)


def compute_ratings(master_df, existing_ratings_df):
    """
    Compute daily Massey power ratings using a season-aware rolling window
    (WINDOW_MULTIPLIER × games-per-team-this-season). Skips dates already
    present in existing_ratings_df. Re-processes the most recent
    RECOMPUTE_TAIL_DAYS ranking_ids each run to absorb late-arriving data.
    """
    max_date_id = max(master_df['grouped_date_id'])
    min_date_id = _MIN_WINDOW
    all_ids = sorted(existing_ratings_df['ranking_id'].unique())
    if len(all_ids) > RECOMPUTE_TAIL_DAYS:
        tail_threshold = all_ids[-RECOMPUTE_TAIL_DAYS]
        n_dropped = int((existing_ratings_df['ranking_id'] >= tail_threshold).sum())
        existing_ratings_df = existing_ratings_df[existing_ratings_df['ranking_id'] < tail_threshold].copy()
        print(f"  Re-processing tail {RECOMPUTE_TAIL_DAYS} game-days "
              f"({n_dropped:,} rows dropped from ratings cache for late-arriving-data refresh)")
    max_ranked = int(max(existing_ratings_df['ranking_id'])) if len(existing_ratings_df) else -1
    min_ranked = int(min(existing_ratings_df['ranking_id'])) if len(existing_ratings_df) else -1

    print("Running DUNCAN ratings for new data...")
    new_frames = []

    # Determine each ranking_id's season once up front so window sizing is fast.
    rid_to_season = (
        master_df.sort_values('grouped_date_id')
                 .drop_duplicates('grouped_date_id', keep='last')
                 .set_index('grouped_date_id')['season']
                 .to_dict()
    )

    for i in range(min_date_id, max_date_id + 1):
        if min_ranked <= i <= max_ranked:
            continue

        season_for_window = rid_to_season.get(i)
        if season_for_window is None:
            prior_ids = [k for k in rid_to_season if k < i]
            season_for_window = rid_to_season[max(prior_ids)] if prior_ids else MIN_SEASON
        window_size = _window_for_season(season_for_window)

        # Don't publish until this season's window can be filled. Each season's
        # window is sized to its game count, so the earliest publishable game-day
        # is the one where the lookback reaches a full window of games.
        if i < window_size:
            continue

        window = master_df[
            (master_df['grouped_date_id'] >= i - (window_size - 1)) &
            (master_df['grouped_date_id'] <= i)
        ].copy()

        window['date_weight'] = (window['grouped_date_id'] - i + window_size) / window_size

        current_date = window['date_game'].max()
        season = window['season'].max()
        print(current_date)

        ranked = _solve_massey(
            window,
            hca=HOME_COURT_ADJUSTMENT,
            weighting_mode=WEIGHTING_MODE,
            margin_transform=MARGIN_TRANSFORM,
            margin_cap=MARGIN_CAP,
        )
        ranked['ranking_date'] = current_date
        ranked['ranking_id'] = i
        ranked['season'] = season
        new_frames.append(ranked)

    ratings_df = pd.concat([existing_ratings_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    ratings_df.sort_values(['ranking_id', 'name'], inplace=True)
    ratings_df.drop_duplicates(keep="first", inplace=True)
    ratings_df['ranking_date'] = pd.to_datetime(ratings_df['ranking_date']).dt.date

    ratings_df.to_csv('duncan_ratings.csv', index=False)
    print("CSV of power rankings is ready!")
    return ratings_df


# =========================================================
# STANDINGS
# =========================================================

def _make_pivot(df, value_col, index_col, new_value_name, aggfunc=np.sum):
    """Helper: pivot, fillna, reset index, and standardize column names."""
    pivot = pd.pivot_table(df, values=value_col, index=[index_col], aggfunc=aggfunc)
    return (
        pivot.fillna(0)
             .reset_index()
             .rename(columns={value_col: new_value_name, index_col: 'name'})
    )


def compute_standings(master_df, existing_standings_df):
    """
    Compute cumulative season standings for each day in master_df.
    Skips dates already present in existing_standings_df.
    """
    # NBA Cup championship games don't count in regular-season W-L by NBA rule —
    # exclude them from the games counted for standings (still in the rating data).
    cols_needed = ['season', 'date_game', 'grouped_date_id', 'visitor_team_name', 'visitor_win', 'home_team_name', 'home_win']
    if 'is_nba_cup_final' in master_df.columns:
        game_df = master_df[master_df['is_nba_cup_final'] != 1][cols_needed]
    else:
        game_df = master_df[cols_needed]
    max_date_id = max(master_df['grouped_date_id'])
    # Standings are cumulative - no window to fill, so we start from the first game-day.
    min_date_id = int(master_df['grouped_date_id'].min())
    if len(existing_standings_df) > 0 and 'ranking_id' in existing_standings_df.columns:
        all_ids = sorted(existing_standings_df['ranking_id'].unique())
        if len(all_ids) > RECOMPUTE_TAIL_DAYS:
            tail_threshold = all_ids[-RECOMPUTE_TAIL_DAYS]
            n_dropped = int((existing_standings_df['ranking_id'] >= tail_threshold).sum())
            existing_standings_df = existing_standings_df[existing_standings_df['ranking_id'] < tail_threshold].copy()
            print(f"  Re-processing tail {RECOMPUTE_TAIL_DAYS} game-days "
                  f"({n_dropped:,} rows dropped from standings cache for late-arriving-data refresh)")
        max_ranked = int(max(existing_standings_df['ranking_id'])) if len(existing_standings_df) else -1
        min_ranked = int(min(existing_standings_df['ranking_id'])) if len(existing_standings_df) else -1
    else:
        max_ranked = -1
        min_ranked = -1

    print("Producing standings...")
    new_frames = []

    for i in range(min_date_id, max_date_id + 1):
        if min_ranked <= i <= max_ranked:
            continue

        season_slice = game_df[game_df['grouped_date_id'] <= i]
        season = season_slice['season'].max()
        season_slice = season_slice[season_slice['season'] == season]
        ranking_date = season_slice['date_game'].max()
        print(ranking_date)

        vw = _make_pivot(season_slice, 'visitor_win', 'visitor_team_name', 'visitor_wins')
        vg = _make_pivot(season_slice, 'visitor_win', 'visitor_team_name', 'visitor_games', aggfunc='count')
        hw = _make_pivot(season_slice, 'home_win',    'home_team_name',    'home_wins')
        hg = _make_pivot(season_slice, 'home_win',    'home_team_name',    'home_games',    aggfunc='count')

        merged = (
            vw.merge(vg, on='name', how='outer')
              .merge(hw, on='name', how='outer')
              .merge(hg, on='name', how='outer')
              .fillna(0)
        )

        merged['wins']   = (merged['visitor_wins'] + merged['home_wins']).astype(int)
        merged['losses'] = (merged['visitor_games'] + merged['home_games'] - merged['wins']).astype(int)
        merged['record'] = merged['wins'].map(str) + "-" + merged['losses'].map(str)
        merged = merged[['name', 'wins', 'losses', 'record']]

        merged['ranking_id']   = i
        merged['ranking_date'] = ranking_date
        merged['season']       = season
        new_frames.append(merged)

    standings_df = pd.concat([existing_standings_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    standings_df.sort_values(['ranking_id', 'name'], inplace=True)
    standings_df.drop_duplicates(keep="first", inplace=True)
    standings_df['ranking_date'] = pd.to_datetime(standings_df['ranking_date']).dt.date

    standings_df.to_csv('daily_standings.csv', index=False)
    print("CSV of standings is ready!")
    return standings_df


# =========================================================
# FINAL ASSEMBLY
# =========================================================

def _get_regular_season_end_date(master_df, season):
    """
    Estimate the last date of the regular season for a given season using
    Option B: find the last date where every active team has played at or
    under the expected regular season game count.
    Falls back to SHORTENED_SEASON_OVERRIDES for lockout/COVID years.
    """
    threshold = REGULAR_SEASON_GAMES.get(int(season), 82)
    season_games = master_df[master_df['season'] == season].copy()

    # Build cumulative game count per team per date
    home = season_games[['date_game', 'home_team_name']].rename(columns={'home_team_name': 'team'})
    away = season_games[['date_game', 'visitor_team_name']].rename(columns={'visitor_team_name': 'team'})
    all_games = pd.concat([home, away]).sort_values('date_game')
    all_games['team_game_num'] = all_games.groupby('team').cumcount() + 1

    # Last date where no team has exceeded the threshold
    within_rs = all_games[all_games['team_game_num'] <= threshold]
    if within_rs.empty:
        return None
    return within_rs['date_game'].max()


def assemble_final(master_df, ratings_df, standings_df):
    """Merge ratings and standings, add flags and last-game context."""
    print("Final step — merging DUNCAN ratings and standings...")

    final_df = pd.merge(ratings_df, standings_df, how='left', on=['ranking_id', 'name'])
    final_df.rename(columns={'ranking_date_x': 'date', 'season_x': 'season'}, inplace=True)
    final_df['season'] = final_df['season'].astype(int)
    final_df['record'] = final_df['record'].fillna("0-0")

    # Flag the most recent date overall
    latest_date_id = final_df['ranking_id'].max()
    final_df['current_date'] = (final_df['ranking_id'] == latest_date_id).astype(int)

    final_df['name_season'] = final_df['name'] + " - " + final_df['season'].map(str)

    # -------------------------------------------------------------------------
    # season_flag: 0 = regular season, 1 = last day of regular season,
    #              2 = last day of postseason
    # -------------------------------------------------------------------------
    final_df['season_flag'] = 0

    # NBA Finals end by late June; season YYYY (basketball-reference convention =
    # season ending in calendar year YYYY) is fully complete after July 31 of that year.
    today = datetime.now().date()
    def season_is_fully_complete(season):
        return today > datetime(int(season), 7, 31).date()

    # Regular season is "done" once any team has played the threshold count.
    # (MIN would break for 2020 bubble — some teams didn't qualify for full schedule.)
    regular_season_complete = set()
    for season in final_df['season'].unique():
        sg = master_df[master_df['season'] == season]
        if sg.empty:
            continue
        home = sg[['home_team_name']].rename(columns={'home_team_name': 'team'})
        away = sg[['visitor_team_name']].rename(columns={'visitor_team_name': 'team'})
        all_g = pd.concat([home, away])
        threshold = REGULAR_SEASON_GAMES.get(int(season), 82)
        if all_g.groupby('team').size().max() >= threshold:
            regular_season_complete.add(season)

    # Last day of postseason — only for seasons where Finals are fully done
    season_max_id = final_df.groupby('season')['ranking_id'].transform('max')
    is_completed = final_df['season'].apply(season_is_fully_complete)
    final_df['season_flag'] = np.where(
        (final_df['ranking_id'] == season_max_id) & is_completed,
        2,
        0
    )

    # Last day of regular season — only for seasons where regular season has actually ended
    for season in final_df['season'].unique():
        if season not in regular_season_complete:
            continue
        rs_end_date = _get_regular_season_end_date(master_df, season)
        if rs_end_date is None:
            continue
        rs_end_str = str(rs_end_date.date()) if hasattr(rs_end_date, 'date') else str(rs_end_date)
        match = final_df[(final_df['season'] == season) & (final_df['date'].astype(str) == rs_end_str)]
        if match.empty:
            continue
        rs_end_ranking_id = match['ranking_id'].max()
        final_df['season_flag'] = np.where(
            (final_df['season'] == season) &
            (final_df['ranking_id'] == rs_end_ranking_id) &
            (final_df['season_flag'] != 2),
            1,
            final_df['season_flag']
        )

    # -------------------------------------------------------------------------
    # Champion & runner-up: detect the Finals series structurally.
    # -------------------------------------------------------------------------
    # NBA Finals = best-of-7 between two specific teams. We declare a champion
    # only when:
    #   1. One team has won 4+ head-to-head games against another within the
    #      last 21 days (the BO7 clinch threshold), AND
    #   2. The last game on file is at least 7 days old. This gates out the
    #      Eastern/Western Conference Finals — also BO7, also clinch at 4 —
    #      which end ~5-10 days before NBA Finals start. Without this gate,
    #      the algorithm would briefly mis-label a conference-final winner as
    #      the league champion in the gap between rounds.
    def detect_finals_champion(season_games):
        sg = season_games.sort_values('date_game')
        if sg.empty:
            return None, None
        last = sg.iloc[-1]
        last_date = pd.to_datetime(last['date_game']).date()
        if (date.today() - last_date).days < 7:
            return None, None
        a = last['home_team_name']
        b = last['visitor_team_name']
        last_dt = pd.Timestamp(last_date)
        window_start = last_dt - pd.Timedelta(days=21)
        sg_dt = pd.to_datetime(sg['date_game'])
        h2h = sg[
            (sg_dt >= window_start) & (sg_dt <= last_dt) &
            (((sg['home_team_name'] == a) & (sg['visitor_team_name'] == b)) |
             ((sg['home_team_name'] == b) & (sg['visitor_team_name'] == a)))
        ]
        a_wins = (((h2h['home_team_name'] == a) & (h2h['home_win'] == 1)) |
                  ((h2h['visitor_team_name'] == a) & (h2h['home_win'] == 0))).sum()
        b_wins = len(h2h) - a_wins
        if a_wins >= 4:
            return a, b
        if b_wins >= 4:
            return b, a
        return None, None

    final_df['champ'] = 0
    final_df['runnerup'] = 0

    for season in final_df['season'].unique():
        season_games = master_df[master_df['season'] == season]
        if season_games.empty:
            continue
        champion, runner_up = detect_finals_champion(season_games)
        if champion is None:
            continue

        champ_season = f"{champion} - {season}"
        runnerup_season = f"{runner_up} - {season}"

        final_df['champ'] = np.where(final_df['name_season'] == champ_season, 1, final_df['champ'])
        final_df['runnerup'] = np.where(final_df['name_season'] == runnerup_season, 1, final_df['runnerup'])

    # Combined status column: 0 = neither, 1 = runner-up, 2 = champion
    final_df['finals_status'] = final_df['runnerup'] + 2 * final_df['champ']

    # -------------------------------------------------------------------------
    # NBA Cup champion & runner-up (since 2023-24)
    # -------------------------------------------------------------------------
    final_df['cup_champ']    = 0
    final_df['cup_runnerup'] = 0

    if 'is_nba_cup_final' in master_df.columns:
        cup_finals = master_df[master_df['is_nba_cup_final'] == 1]
        for _, game in cup_finals.iterrows():
            if game['home_win'] == 1:
                champion, runner_up = game['home_team_name'], game['visitor_team_name']
            else:
                champion, runner_up = game['visitor_team_name'], game['home_team_name']
            champ_ns    = f"{champion} - {game['season']}"
            runnerup_ns = f"{runner_up} - {game['season']}"
            final_df['cup_champ']    = np.where(final_df['name_season'] == champ_ns,    1, final_df['cup_champ'])
            final_df['cup_runnerup'] = np.where(final_df['name_season'] == runnerup_ns, 1, final_df['cup_runnerup'])

    # 0 = neither, 1 = cup runner-up, 2 = cup champion
    final_df['cup_status'] = final_df['cup_runnerup'] + 2 * final_df['cup_champ']

    # -------------------------------------------------------------------------
    # Last game result
    # -------------------------------------------------------------------------
    final_df['date_str'] = final_df['date'].astype(str)

    lastgameh = (
        master_df[['date_game', 'home_team_name', 'home_result', 'visitor_team_name']]
        .rename(columns={'home_team_name': 'name', 'date_game': 'date_str'})
        .assign(date_str=lambda d: d['date_str'].astype(str))
    )
    lastgamev = (
        master_df[['date_game', 'visitor_team_name', 'visitor_result', 'home_team_name']]
        .rename(columns={'visitor_team_name': 'name', 'date_game': 'date_str'})
        .assign(date_str=lambda d: d['date_str'].astype(str))
    )

    final_df = final_df.merge(lastgameh, how='left', on=['date_str', 'name'])
    final_df = final_df.merge(lastgamev, how='left', on=['date_str', 'name'])

    for col in ['home_result', 'visitor_result', 'home_team_name', 'visitor_team_name']:
        final_df[col] = final_df[col].fillna("")

    final_df['last_game_result'] = (final_df['home_result'] + final_df['visitor_result'])
    final_df['opponent'] = final_df['home_team_name'] + final_df['visitor_team_name']

    final_df = final_df[[
        'ranking_id', 'date', 'season', 'name', 'rating', 'rank',
        'record', 'current_date', 'season_flag', 'name_season',
        'champ', 'runnerup', 'finals_status',
        'cup_champ', 'cup_runnerup', 'cup_status',
        'last_game_result', 'opponent'
    ]]

    final_df.to_csv('duncan_ratings_with_standings.csv', index=False)
    print("CSV of everything is ready!")
    return final_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    max_season = datetime.now().year + 1

    # 1. Scrape
    existing_games = pd.read_csv("loaded_nba_games.csv")
    raw_df = scrape_games(MIN_SEASON, max_season, existing_games)

    # 2. Prepare game data
    master_df = prepare_game_data(raw_df)

    # 3. Ratings
    existing_ratings = pd.read_csv("duncan_ratings.csv")
    ratings_df = compute_ratings(master_df, existing_ratings)

    # 4. Standings
    try:
        existing_standings = pd.read_csv("daily_standings.csv")
    except FileNotFoundError:
        existing_standings = pd.DataFrame(columns=['name', 'wins', 'losses', 'record', 'ranking_id', 'ranking_date', 'season'])
    standings_df = compute_standings(master_df, existing_standings)

    # 5. Final merge
    assemble_final(master_df, ratings_df, standings_df)
