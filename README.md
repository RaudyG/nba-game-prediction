# NBA Game-Outcome Prediction

An end-to-end machine learning pipeline that predicts the winner of an NBA game from each team's recent form. Built with a focus on the engineering decisions that separate a model you can trust from one you can't: **leakage control, honest baselines, and time-aware validation.**

> **Result:** 61.3% accuracy vs a 50% baseline (+11.3 points) on ~2,500 games, using time-series cross-validation. Public NBA models top out around 65–70%, so this sits in a realistic range — not an inflated one.

## Why this project

Predicting sports outcomes is a clean way to demonstrate the parts of ML that actually matter in production: acquiring messy data, engineering features without leaking the future, and validating honestly. The same principles transfer directly to problems like credit-default prediction in fintech.

## Pipeline overview

| Stage | What it does |
|-------|--------------|
| `src/scraper.py` | Pulls regular-season + playoff game logs for all 30 NBA teams from the official stats API and writes a single stacked dataset. |
| `src/model.py` | Builds leakage-safe rolling features, engineers opponent-relative differences, trains a regularized logistic regression, and evaluates with time-series CV. |
| `src/predict.py` | Trains on the full league and outputs a win probability for a specific matchup. |

## Key engineering decisions

**1. Leakage control.** Every feature is computed *only* from games prior to the one being predicted — lagged 10-game rolling averages (`shift(1)` before `rolling()`). Using a game's own box score to predict that game is the most common, and most invisible, way to ship a model that scores 95% in testing and fails in production.

**2. Opponent-relative features.** Instead of a team's absolute form, each matchup is modeled as the *difference* between the two teams' recent form (joined on `Game_ID`). What predicts an outcome isn't your raw shooting % — it's how much better or worse you are than tonight's opponent. This also makes the dataset symmetric, giving a clean 50% baseline.

**3. Honest validation.** Time-series cross-validation (`TimeSeriesSplit`) never trains on the future to predict the past. The model is reported against the baseline of random guessing, not in isolation.

**4. Sample-size awareness.** An early version trained on just 2 teams (~150 games) and learned noise — feature coefficients flipped sign depending on the rolling window. Scaling to all 30 teams (~2,600 games) let the model learn the *general* rule of what wins a basketball game, which then applies to any specific matchup.

## Most influential features

Trained on the full league, the coefficients are physically sensible and stable in sign:

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `winpct_form_diff` | +0.58 | Recent win-rate edge over opponent — the strongest signal |
| `REB_form_diff` | +0.23 | Rebounding edge |
| `FG_PCT_form_diff` | +0.23 | Shooting-efficiency edge |
| `is_home` | +0.22 | Home-court advantage |
| `TOV_form_diff` | −0.22 | Turning the ball over more than the opponent hurts |

## A note on honesty

When evaluated *only* on games involving two strong teams, the model does **not** beat the baseline — because for two winning teams, "always predict a win" is already ~70% accurate. The +11.3-point edge holds across the league as a whole, where outcomes are balanced. Reporting this matters more than hiding it: a model's value depends entirely on the context it's measured in.

## Quick start

```bash
pip install -r requirements.txt

# Option A: use the included dataset (data/nba_2025_26_games.xlsx)
cd src
python model.py        # train + evaluate
python predict.py      # win probability for a matchup

# Option B: regenerate the dataset yourself
python scraper.py      # ~2-3 min (rate-limited API calls)
```

## Tech stack

Python · pandas · scikit-learn · nba_api

## Limitations & next steps

- Features are based mostly on regular-season data; playoff dynamics (rotations, matchups, momentum) differ and aren't fully captured.
- Single linear model by design — interpretability over raw performance at this sample size.
- **Next:** weight playoff games more heavily, add an explicit opponent-quality feature, and benchmark against a gradient-boosted model once the dataset spans multiple seasons.

---

*Built as a portfolio project exploring rigorous, leakage-free ML — the same discipline applied to sports as to financial risk modeling.*
