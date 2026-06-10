import pandas as pd, numpy as np
import model as m

df = m.load(m.DATA_FILE)
df = m.add_rolling(df)
merged, feat_cols = m.make_relative(df)

# Train on the entire league
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3))
model.fit(merged[feat_cols].values, merged["win"].values)

# Most recent form of each team (last available record)
def latest_form(team):
    sub = df[df["TEAM"]==team].sort_values("GAME_DATE")
    return sub.iloc[-1]

nyk = latest_form("NYK")
sas = latest_form("SAS")

form_cols = [f"{s}_form" for s in m.STATS] + ["winpct_form", "rest_days"]

# Build the prediction row: NYK as home (Game 4 is at MSG)
# relative features = NYK form - SAS form
row = {}
for c in form_cols:
    row[f"{c}_diff"] = nyk[c] - sas[c]
row["is_home"] = 1  # Game 4 in New York

X_pred = pd.DataFrame([row])[feat_cols].values
prob_nyk = model.predict_proba(X_pred)[0][1]

print("="*55)
print("GAME 4 PREDICTION - NYK (home) vs SAS")
print("="*55)
print(f"  Recent form NYK (win% last 10): {nyk['winpct_form']:.1%}")
print(f"  Recent form SAS (win% last 10): {sas['winpct_form']:.1%}")
print(f"  Rest NYK: {nyk['rest_days']:.0f}d | SAS: {sas['rest_days']:.0f}d")
print()
print(f"  >> NYK win probability: {prob_nyk:.1%}")
print(f"  >> SAS win probability: {1-prob_nyk:.1%}")
