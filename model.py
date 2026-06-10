"""
Predictive model for NBA game results (win/loss).
Trains with the 30 teams (../data/nba_2025_26_games.xlsx, stacked sheet) to learn
the GENERAL rule of what makes teams win, and evaluates the prediction.

METHODOLOGICAL KEY:
1. Anti-leakage: features with games BEFORE ONLY (shift + rolling).
2. RELATIVE features to opponent: for each game, the team's form is crossed with
   that of its opponent (via Game_ID). This is what really predicts: your absolute
   FG% doesn't matter, but how much better/worse than the opponent.
3. Strong regularization (low C) because even with 2600 rows it's advisable.

Requirements: pip install pandas scikit-learn openpyxl
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(_HERE, "..", "data", "nba_2025_26_games.xlsx")
ROLL_WINDOW = 10        # recent form: last 10 games
# Statistics to average (those with predictive value)
STATS = ["PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "OREB", "AST", "TOV",
         "STL", "BLK"]


def load(path):
    df = pd.read_excel(path, sheet_name=0)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y",
                                     errors="coerce")
    df["win"] = df["WL"].astype(str).str.startswith("W").astype(int)
    df["is_home"] = (~df["MATCHUP"].astype(str).str.contains("@")).astype(int)
    return df.sort_values(["TEAM", "GAME_DATE"]).reset_index(drop=True)


def add_rolling(df):
    """Moving averages ONLY from previous games (shift(1) = anti-leakage)."""
    df = df.copy()
    # recent margin of victory and rest
    df["rest_days"] = (
        df.groupby("TEAM")["GAME_DATE"].diff().dt.days.fillna(3).clip(upper=10)
    )
    for stat in STATS:
        df[f"{stat}_form"] = (
            df.groupby("TEAM")[stat]
            .transform(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=3).mean())
        )
    df["winpct_form"] = (
        df.groupby("TEAM")["win"]
        .transform(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=3).mean())
    )
    return df


def make_relative(df):
    """
    Crosses each game with itself by Game_ID to get RELATIVE features
    (team - opponent). Each Game_ID has exactly 2 rows.
    Returns a dataset with one row per (team, game) and *_diff columns.
    """
    form_cols = [f"{s}_form" for s in STATS] + ["winpct_form", "rest_days"]
    keep = ["Game_ID", "TEAM", "GAME_DATE", "win", "is_home", "is_playoff"] + form_cols
    d = df[keep].dropna(subset=form_cols).copy()

    # Join each row with the opponent's row in the same Game_ID
    merged = d.merge(d, on="Game_ID", suffixes=("", "_opp"))
    merged = merged[merged["TEAM"] != merged["TEAM_opp"]].copy()

    # Relative features: my form minus the opponent's form
    for c in form_cols:
        merged[f"{c}_diff"] = merged[c] - merged[f"{c}_opp"]
    # is_home is already absolute (1 if home game)
    feat_cols = [f"{c}_diff" for c in form_cols] + ["is_home"]
    return merged, feat_cols


def evaluate(merged, feat_cols, label, mask=None):
    sub = merged if mask is None else merged[mask]
    X = sub[feat_cols].values
    y = sub["win"].values
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.3),  # low C = strong regularization
    )
    cv = TimeSeriesSplit(n_splits=5)
    # sort by date so the temporal split is valid
    order = sub["GAME_DATE"].argsort().values
    scores = cross_val_score(model, X[order], y[order], cv=cv, scoring="accuracy")
    baseline = max(y.mean(), 1 - y.mean())
    print(f"\n--- {label} ---")
    print(f"  Games:              {len(y)}")
    print(f"  Baseline:              {baseline:.1%}")
    print(f"  Model accuracy:       {scores.mean():.1%} (+/- {scores.std():.1%})")
    print(f"  Improvement over baseline: {scores.mean() - baseline:+.1%}")
    return model


def main():
    df = load(DATA_FILE)
    df = add_rolling(df)
    merged, feat_cols = make_relative(df)
    print(f"Dataset: {len(merged)} filas (equipo-partido), {len(feat_cols)} features relativos")
    print("=" * 55)

    # 1) Evaluar con TODA la liga (aprende la regla general)
    model = evaluate(merged, feat_cols, "TODA LA LIGA (30 equipos)")

    # 2) Evaluar solo en partidos de NYK y SAS (lo que te interesa)
    mask = merged["TEAM"].isin(["NYK", "SAS"])
    evaluate(merged, feat_cols, "Only NYK and SAS games", mask=mask)

    # 3) Train on the entire league and see what features matter
    X = merged[feat_cols].values
    y = merged["win"].values
    model.fit(X, y)
    coefs = model.named_steps["logisticregression"].coef_[0]
    imp = sorted(zip(feat_cols, coefs), key=lambda x: abs(x[1]), reverse=True)
    print("\n" + "=" * 55)
    print("Most influential features (trained on entire league):")
    for name, c in imp[:8]:
        s = "favors winning" if c > 0 else "favors losing"
        print(f"  {name:22s} {c:+.3f}  ({s})")

    return model, merged, feat_cols


if __name__ == "__main__":
    main()
