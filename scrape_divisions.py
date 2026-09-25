"""One-off: NBA division membership per season, 1977-2015, from
basketball-reference standings pages -> nba_divisions.csv.

Only needed for seeding in the title-odds sim: division winners were
guaranteed top playoff seeds through 2015. Static history; rerun only if
team names change in DUNCAN's data.
"""
import re, time
from io import StringIO
import pandas as pd
import requests

rows = []
for year in range(1977, 2016):
    r = requests.get(f'https://www.basketball-reference.com/leagues/NBA_{year}_standings.html',
                     headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
    r.raise_for_status()
    html = r.text.replace('<!--', '').replace('-->', '')
    for t in pd.read_html(StringIO(html)):
        first = str(t.columns[0])
        if 'Conference' not in first:
            continue
        conf = 'East' if 'Eastern' in first else 'West'
        div = None
        for cell in t.iloc[:, 0].astype(str):
            if cell.endswith('Division'):
                div = cell.replace(' Division', '')
                continue
            team = re.sub(r'\*|\s*\(\d+\)$', '', cell).strip()
            if div and team and team != 'nan':
                rows.append((year, team, conf, div))
    print(year, end=' ', flush=True)
    time.sleep(3.5)
from duncan import TEAM_ALIASES  # DUNCAN names each franchise once
rows = [(y, TEAM_ALIASES.get(t, t), c, v) for y, t, c, v in rows]
df = pd.DataFrame(rows, columns=['season', 'team', 'conference', 'division']).drop_duplicates(['season', 'team'])
df.to_csv('nba_divisions.csv', index=False)
print(f'\n{len(df)} team-seasons')
