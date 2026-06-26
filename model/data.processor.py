import pandas as pd
import os

TEAMS_FOLDER = r"C:\licenta\football-predictor\model\teams"
SAVE_PATH = r"C:\licenta\football-predictor\model"

def compute_features(prev_matches, role):
    if len(prev_matches) < 5:
        return None

    m = prev_matches.sort_values("date", ascending=False)
    last5 = m.head(5)

    features = {
        f"{role}_goals_avg": m["goals_scored"].mean(),
        f"{role}_conceded_avg": m["goals_conceded"].mean(),
        f"{role}_goal_diff_avg": (m["goals_scored"] - m["goals_conceded"]).mean(),
        f"{role}_home_win_rate": (m[m["is_home"] == True]["result"] == "W").mean() if any(m["is_home"]) else 0.5,
        f"{role}_away_win_rate": (m[m["is_home"] == False]["result"] == "W").mean() if not all(m["is_home"]) else 0.5,
        #optionale
        f"{role}_form_pts_last5": last5["result"].map({"W": 3, "D": 1, "L": 0}).mean(),
        f"{role}_clean_sheets_rate": (m["goals_conceded"] == 0).mean(),
        f"{role}_win_streak": (last5["result"] == "W").sum(),
        f"{role}_loss_streak": (last5["result"] == "L").sum(),
        f"{role}_draw_rate": (m["result"] == "D").mean(),
        f"{role}_avg_total_goals": (m["goals_scored"] + m["goals_conceded"]).mean(),
        f"{role}_over_2_5_rate": ((m["goals_scored"] + m["goals_conceded"]) > 2.5).mean(),
        f"{role}_btts_rate": ((m["goals_scored"] > 0) & (m["goals_conceded"] > 0)).mean(),
        f"{role}_scoring_consistency": (m["goals_scored"] > 0).mean(),
        f"{role}_defense_strength": 1 / (m["goals_conceded"].mean() + 1)
    }
    return features

all_dfs = [pd.read_csv(os.path.join(TEAMS_FOLDER, f)) for f in os.listdir(TEAMS_FOLDER) if
           f.endswith(".csv") and f != "teams_index.csv"]
all_matches = pd.concat(all_dfs).reset_index(drop=True)
all_matches["date"] = pd.to_datetime(all_matches["date"]).dt.tz_localize(None)
all_matches = all_matches.sort_values("date").drop_duplicates(subset=["fixture_id", "team_id"])

home_df = all_matches[all_matches["is_home"] == True].copy()
away_df = all_matches[all_matches["is_home"] == False].copy()
matches = pd.merge(home_df[["fixture_id", "date", "team_id", "result"]], away_df[["fixture_id", "team_id"]],
                   on="fixture_id", suffixes=("_home", "_away"))

grouped = {tid: grp for tid, grp in all_matches.groupby("team_id")}
rows = []

for _, match in matches.iterrows():
    h_prev = grouped[match["team_id_home"]][grouped[match["team_id_home"]]["date"] < match["date"]]
    a_prev = grouped[match["team_id_away"]][grouped[match["team_id_away"]]["date"] < match["date"]]

    if len(h_prev) >= 5 and len(a_prev) >= 5:
        h_feats = compute_features(h_prev, "home")
        a_feats = compute_features(a_prev, "away")
        if h_feats and a_feats:
            rows.append({**h_feats, **a_feats, "result": match["result"], "home_team_id": match["team_id_home"],   "away_team_id": match["team_id_away"]})

df_master = pd.DataFrame(rows).fillna(0)
df_master.to_csv(os.path.join(SAVE_PATH, "master_features.csv"), index=False)
print(f"Succes! Am generat dataset-ul cu {df_master.shape[0]} meciuri și {df_master.shape[1] - 1} coloane.")