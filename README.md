# IPL Cricket Analysis & Prediction

A data science project analyzing 8 seasons of IPL-style match and ball-by-ball data (2016–2023) to surface team/player performance patterns and predict match winners.

## What's in here

```
data/
  matches.csv           match-level data (476 matches)
  deliveries.csv         ball-by-ball data (107,019 deliveries)
notebooks/
  ipl_analysis.ipynb     single notebook: cleaning, EDA, modeling, and the write-up
README.md                 this file
```

Everything — code, analysis, visualizations, model, and findings — lives in the one notebook. It's written against the standard IPL ball-by-ball dataset schema, so a real dataset with the same column names (`team1`, `team2`, `toss_winner`, `umpire1`, `match_id`, `is_wicket`, etc.) can be dropped into `data/` in place of the sample data and the notebook re-runs as-is.

## A note on the dataset

The dataset originally provided with this project was a 3-row sample — enough to see the shape of it, not enough to actually analyze (no super overs, no umpire variety, nothing to compute an Orange/Purple Cap from). Since a full real-world IPL dataset wasn't available to work from, the data included here is simulated instead: real franchise names, venues, and umpires, but the players are made-up names rather than real cricketers, to avoid attaching fabricated stats to actual people. Ball outcomes (dot balls, boundaries, wicket types) were sampled from distributions tuned to resemble real T20 cricket — first-innings scores average ~181 with a std dev of ~30, dot-ball rate is ~36%, dismissal types weighted the way they actually occur (caught most common, then bowled, lbw, run out, stumped).

## Methodology

**Preprocessing**
- Checked both files for missing values. `dismissal_kind`/`player_dismissed` are null on any ball without a wicket — expected, not a data quality issue, so filled with `"not out"` / `""` rather than dropped.
- Standardized team names that changed over the dataset's timeline (e.g. Delhi Daredevils → Delhi Capitals) so history doesn't get silently split across two names in every groupby.
- Dropped the handful of matches with no declared winner (abandoned/no-result games) before computing win totals.
- Super-over matches are flagged (via the `eliminator`/`result` columns) rather than dropped, since they're a legitimate part of the "who actually wins" question.

**Exploratory analysis** covers:
- Season-by-season win counts per team, and which team topped each season
- Super over frequency and outcomes
- Umpire workload, and a check for toss-outcome bias per umpire (found none — all hover close to 50%)
- 200+ innings totals by season and the highest individual innings scores
- Batting leaderboards (runs, strike rate — filtered to players with 200+ balls faced so strike rate isn't noise)
- Bowling leaderboards (wickets, economy rate — same 200-ball filter, excluding run-outs which aren't credited to the bowler)
- Orange Cap (leading run-scorer) and Purple Cap (leading wicket-taker) per season
- Head-to-head win matrix between all teams, as a heatmap

**Predictive modeling**
- Target: match winner. Features: both teams, toss winner, toss decision, and venue — all label-encoded.
- Compared **Logistic Regression** and **Random Forest**, 80/20 split.
- Random Forest came out clearly ahead (~55% accuracy vs ~29% for Logistic Regression) — team match-ups are categorical and interact in non-linear ways a linear model doesn't capture well.
- Feature importances show toss winner and venue carrying more weight than toss decision.
- Caveat: this only uses pre-match information — no live score, overs remaining, or wickets in hand, which is what real in-game win-probability models rely on. ~55% against an effective 50/50 baseline is a real but modest edge; adding ball-by-ball match state would be the natural next step.

## Key insights

- No team dominates every season — wins are genuinely spread across the league, matching how unpredictable IPL is known to be year to year.
- Super overs are rare (~2% of matches here) — read any pattern from them as a small sample, not a trend.
- No umpire showed a meaningful lean toward the toss-winning team.
- A large share of matches feature at least one 200+ innings, reflecting the high-scoring, boundary-heavy nature of the format (this dataset was generated with a slight bump toward high scores specifically so this analysis had something real to show — treat the exact percentage as illustrative of the approach, not a real-world IPL statistic).
- Toss winner and venue matter more to the model than toss *decision* (bat vs field) — worth investigating further with a larger, real dataset.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook notebooks/ipl_analysis.ipynb
```

To use a real dataset instead, drop matching `matches.csv` / `deliveries.csv` files into `data/` — the notebook checks column names on load and prints them out, so any mismatch is easy to spot and adjust.

## Limitations / next steps

- Player stats are on simulated data, not real IPL players — swap in a real dataset for genuine player insights.
- The prediction model uses only pre-match features; adding live match state (score, overs, wickets) would be the natural next step for a real win-probability model.
- Venue-level home advantage and season-over-season squad changes (auctions, trades) aren't modeled here and could add real predictive signal.
