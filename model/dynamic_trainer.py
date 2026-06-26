import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

MASTER_PATH = r"C:\football-predictor\model\master_features.csv"
TEAMS_FOLDER = r"C:\football-predictor\backend\teams"


def compute_confidence(probs, num_matches):

    model_conf = max(probs)  # ex: 0.65


    if num_matches >= 100:
        data_quality = 1.0
    elif num_matches >= 70:
        data_quality = 0.7
    elif num_matches >= 38:
        data_quality = 0.4
    else:
        data_quality = 0.1


    final_score = (model_conf * 0.6) + (data_quality * 0.4)

    if final_score >= 0.70:
        label = "Ridicată"
    elif final_score >= 0.50:
        label = "Medie"
    else:
        label = "Scăzută"

    return round(final_score * 100, 1), label
#pentru un meci gen city brighton in care city e favorita deci modelconf mare si ambele echipe au 114 meciuri = incredere mare
#exemplu pentru meci scazut ar fi auxxere ajaccio pentru ca e meci echilibrat si au si putine meciuri


def get_num_matches(team_name):
    for f in os.listdir(TEAMS_FOLDER):
        if team_name.replace(" ", "_") in f:
            team_df = pd.read_csv(os.path.join(TEAMS_FOLDER, f))
            return len(team_df)
    return 0 # pentru ca meciurile din master sunt mai putine decat in teams(le am filtrat)

def get_team_id(team_name):
        for f in os.listdir(TEAMS_FOLDER):
            if team_name.replace(" ", "_") in f:

                return int(f.split("_")[0])
        raise ValueError(f"Echipa negăsită: {team_name}")

def get_pie_chart_prediction(home_team, away_team, selected_optionals):
    df = pd.read_csv(MASTER_PATH)

    base_cols = [
        "home_goals_avg", "home_conceded_avg", "home_goal_diff_avg",
        "home_home_win_rate", "home_away_win_rate",
        "away_goals_avg", "away_conceded_avg", "away_goal_diff_avg",
        "away_home_win_rate", "away_away_win_rate"
    ]

    # 2. Adăugare opționale
    final_features = base_cols.copy()
    for opt in selected_optionals:
        final_features.extend([f"home_{opt}", f"away_{opt}"])

    X = df[final_features]
    y = df["result"]

    # 3. Antrenare
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    #
    home_id = get_team_id(home_team)
    away_id = get_team_id(away_team)


    home_stats = df[df["home_team_id"] == home_id].iloc[-1]
    away_stats = df[df["away_team_id"] == away_id].iloc[-1]

    match_row = {}
    for col in final_features:
        if col.startswith("home_"):
            match_row[col] = home_stats[col]
        elif col.startswith("away_"):
            match_row[col] = away_stats[col]

    current_match_features = pd.DataFrame([match_row])[final_features]

    # 5. Predicție
    probs = model.predict_proba(current_match_features)[0]
    classes = model.classes_  # ['D', 'L', 'W']
    prob_map = dict(zip(classes, probs))

    num_matches_home = get_num_matches(home_team)
    num_matches_away = get_num_matches(away_team)
    num_matches = min(num_matches_home,num_matches_away)
    confidence_score, confidence_label = compute_confidence(probs, num_matches)

    win = round(prob_map.get('W', 0) * 100, 1)
    draw = round(prob_map.get('D', 0) * 100, 1)
    lose = round(prob_map.get('L', 0) * 100, 1)

    labels = ["Victorie Gazde (W)", "Egal (D)", "Victorie Oaspeți (L)"]
    data_values = [win, draw, lose]
    colors = ["#4CAF50", "#FFC107", "#F44336"]

    response_data = {
        "pie": {
            "labels": labels,
            "datasets": [{
                "data": data_values,
                "backgroundColor": colors
            }],

        },

        "bar": {
            "labels": labels,
            "datasets": [{
                "label": "Probabilitate (%)",
                "data": data_values,
                "backgroundColor": colors,
                "borderWidth": 1
            }],

        },
        "confidence_score": confidence_score,
        "confidence_label": confidence_label
    }

    return response_data

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import cross_val_score, GridSearchCV

    df = pd.read_csv(MASTER_PATH)

    # Toate coloanele (base + optionale)
    all_cols = [
        # Base
        "home_goals_avg", "home_conceded_avg", "home_goal_diff_avg",
        "home_home_win_rate", "home_away_win_rate",
        "away_goals_avg", "away_conceded_avg", "away_goal_diff_avg",
        "away_home_win_rate", "away_away_win_rate",
        # Optionale
        "home_form_pts_last5", "away_form_pts_last5",
        "home_clean_sheets_rate", "away_clean_sheets_rate",
        "home_win_streak", "away_win_streak",
        "home_loss_streak", "away_loss_streak",
        "home_draw_rate", "away_draw_rate",
        "home_avg_total_goals", "away_avg_total_goals",
        "home_over_2_5_rate", "away_over_2_5_rate",
        "home_btts_rate", "away_btts_rate",
        "home_scoring_consistency", "away_scoring_consistency",
        "home_defense_strength", "away_defense_strength"
    ]

    X = df[all_cols]
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )


    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "class_weight": ["balanced"]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("=== GRID SEARCH ===")
    print(f"Cei mai buni parametri sugerati: {grid.best_params_}")
    print(f"Cea mai buna acuratete CV: {round(grid.best_score_ * 100, 2)}%")


    grid_model = grid.best_estimator_
    grid_train = accuracy_score(y_train, grid_model.predict(X_train))
    grid_test = accuracy_score(y_test, grid_model.predict(X_test))
    print(f"  -> Train: {round(grid_train * 100, 2)}% | Test: {round(grid_test * 100, 2)}% "
          f"(diferenta: {round((grid_train - grid_test) * 100, 2)}%)")


    model = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")


    print(f"Train Accuracy:      {round(accuracy_score(y_train, y_train_pred) * 100, 2)}%")
    print(f"Test Accuracy:       {round(accuracy_score(y_test, y_test_pred) * 100, 2)}%")
    print(f"Cross Val (5-fold):  {round(cv_scores.mean() * 100, 2)}% ± {round(cv_scores.std() * 100, 2)}%")


    print(classification_report(y_test, y_test_pred,
                                target_names=["Egal (D)", "Victorie Oaspeți (L)", "Victorie Gazde (W)"]))