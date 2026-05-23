# DUNCAN Predictive Analysis: How Well Do the Ratings Predict NBA Outcomes?

**Date of analysis:** 2026-05-22
**Window:** 2020-2026 seasons (8,265 regular-season + playoff games with opening Vegas spreads)
**Question:** Do DUNCAN's WLS ratings have any predictive edge, especially relative to Vegas point spreads?

---

## The setup

DUNCAN produces a daily rating per team where the units are roughly "points vs an average team." That makes it natural to derive a predicted game margin:

```
predicted_home_margin = home_rating - away_rating + 2.0 (home court adjustment)
```

For each game in our window we compared this predicted margin against:

1. **Vegas's opening spread** (the market's predicted margin)
2. **The actual final margin** of the game

Spreads sourced from the open-source NBA_Betting project (SQLite snapshot, January 2026). Each team's "pre-game rating" is the snapshot taken the night before the game (so the rating cannot peek at the game's own result).

---

## Regular-season results

| Metric | DUNCAN | Vegas |
|---|---|---|
| Margin RMSE | 13.92 pts | 13.39 pts |
| Margin MAE | 10.94 pts | 10.44 pts |
| Mean error (home bias) | -0.01 | +0.08 |
| Straight-up win prediction | **64.3%** | n/a |
| ATS (pushes excluded) | **49.3%** | n/a (it's the market) |

**Takeaways:**

- DUNCAN trails Vegas by ~0.5 RMSE every single season. That gap is small but consistent, which is what you'd expect from a rating that doesn't see injuries, lineups, or rest decisions.
- 64.3% straight-up is genuinely strong (well above the 55.3% home-win base rate). The ratings know who's good.
- 49.3% ATS is below 50%, and well below the 52.38% break-even at standard -110 juice. **Betting DUNCAN against the spread would lose money.**

### Calibration: where DUNCAN systematically misses

| DUNCAN predicted | Actual mean | Residual |
|---|---|---|
| under -15 (heavy road favorite) | -11.79 | **+4.85** |
| -15 to -10 | -10.44 | +1.36 |
| -3 to +3 (close games) | matches Vegas closely | ~0 |
| +10 to +15 | +11.18 | -0.86 |
| over +15 (heavy home favorite) | +13.88 | **-3.20** |

**The model over-extrapolates at the extremes.** When DUNCAN says a team is a 20-point favorite, the reality is more like a 14-point favorite. The MARGIN_CAP=25 in training protects the rating math, but the predicted-margin output still tails off vs reality. This is the cleanest "actionable" finding from the whole exercise.

---

## Playoffs (n=527 games)

The hypothesis going in: load management and rotation noise should clear up in the playoffs, so DUNCAN should be relatively closer to Vegas.

| Metric | Reg season | Playoffs |
|---|---|---|
| DUNCAN - Vegas RMSE gap | +0.55 | +0.36 |
| DUNCAN SU accuracy | 64.4% | 62.2% |
| DUNCAN ATS | 49.4% | 47.1% |

DUNCAN does close the gap to Vegas slightly in playoffs (mild support for the load-management hypothesis), but absolute variance is higher in playoffs for both models, and ATS is actually worse.

---

## Series-level prediction (n=84 series, 2020-2025)

This was the most interesting cut. Rather than asking "did DUNCAN call the spread right," we asked "did DUNCAN call the series winner right."

| Predictor | Series accuracy |
|---|---|
| DUNCAN (higher pre-series rating) | **65.5%** |
| Vegas (Game 1 favorite) | **66.7%** |

**DUNCAN essentially matches Vegas at predicting series winners.**

When the two methods agreed (63 series), they were right 71.4% of the time. When they disagreed (21 series), DUNCAN's pick won 47.6%, Vegas's pick won 52.4% — a wash. Neither has a real edge over the other when they see the matchup differently.

### Where DUNCAN's confidence translates to accuracy

| Pre-series rating gap | n | DUNCAN accuracy |
|---|---|---|
| under 1.5 pts (toss-ups) | 33 | 51.5% (coin flip) |
| 1.5 - 3 pts | 14 | 78.6% |
| 3 - 5 pts | 24 | 66.7% |
| 5 - 7 pts | 6 | 83.3% |
| over 7 pts | 7 | 85.7% |

When DUNCAN says "this is a real mismatch," it's right ~80% of the time. When it says "toss-up," it's truly a toss-up.

### DUNCAN's correct underdog calls

DUNCAN picked the Vegas underdog and was right on 10 series, including:
- 2024 Minnesota over Denver (defending champ)
- 2024 + 2025 Indiana over New York
- 2025 Minnesota over LA Lakers
- 2023 LA Lakers over Golden State
- 2022 Boston over Miami

### DUNCAN's biggest series misses

The model was most overconfidently wrong in famous-upset series:
- 2020 Bubble: Clippers blew a 3-1 lead to Denver
- 2022 Finals: Golden State beat Boston
- 2025: Knicks beat Celtics, Pacers beat Cavaliers

These are upsets in the public memory, so DUNCAN's "wrong" picks line up with what surprised the basketball world too.

---

## What we explored but didn't finish

**Series-price ROI:** the natural next question was "if you bet DUNCAN's series pick at the Vegas series moneyline every round, do you end up ahead?" Historical series moneylines aren't openly archived (we'd have had to derive them from Game 1 spreads using best-of-7 math). Given the per-game ATS already came in at 49.3%, the directional answer is "behind," and the user opted not to pursue.

---

## Bottom line

The DUNCAN ratings have **real predictive power for who wins** (64% straight-up regular season, 65.5% series winners). They are **not a market-beater** for point spreads (49.3% ATS, no edge bucket clears break-even).

This makes sense: a team-strength rating from game results alone cannot beat a market that also incorporates injuries, lineups, rest, and news.

### Site-feature shapes that ARE viable

1. **"Tonight's matchup preview"** showing the DUNCAN-predicted spread and SU favorite, framed as transparency rather than a bet.
2. **"Playoff Predictions board"** showing pre-series ratings, DUNCAN's predicted winner, and actual outcomes. 65.5% accuracy is honest, defensible content.
3. **"Upset board"** highlighting the 10 series where DUNCAN picked the underdog and was right.

### Site-feature shapes that are NOT viable

Anything framed as "beat the spread" or "find Vegas's mistakes." The data does not support those claims.

---

## Artifacts

All in `NBA/predictive_analysis/`:

- `build_dataset.py` - joins games + ratings + spreads into a per-game dataset
- `analyze.py` - regular-season and per-game playoff metrics
- `series_analysis.py` - series-level winner accuracy
- `dataset.csv` - 8,265 game rows (game, ratings, spread, actual margin, ATS result)
- `series.csv` - 84 series rows with pre-series ratings, predicted winner, actual outcome
