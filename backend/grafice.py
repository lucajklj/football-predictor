import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

MASTER_PATH = r"C:\licenta\football-predictor\model\master_features.csv"

df = pd.read_csv(MASTER_PATH)

base_cols = [
    "home_goals_avg", "home_conceded_avg", "home_goal_diff_avg",
    "home_home_win_rate", "home_away_win_rate",
    "away_goals_avg", "away_conceded_avg", "away_goal_diff_avg",
    "away_home_win_rate", "away_away_win_rate"
]

all_cols = base_cols + [
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



depths = range(2, 16)
train_scores = []
cv_scores_list = []

for d in depths:
    m = RandomForestClassifier(n_estimators=200, max_depth=d,
                               class_weight="balanced", random_state=42)
    # Train accuracy (pe tot setul)
    m.fit(X, y)
    train_scores.append(accuracy_score(y, m.predict(X)) * 100)
    # Cross-validation 5-fold
    cv = cross_val_score(m, X, y, cv=5, scoring="accuracy")
    cv_scores_list.append(cv.mean() * 100)

plt.figure(figsize=(9, 5))
plt.plot(list(depths), train_scores, marker="o", label="Acuratețe antrenare", color="#1a73e8")
plt.plot(list(depths), cv_scores_list, marker="s", label="Acuratețe validare încrucișată (5-fold)", color="#e8731a")
plt.axvline(x=5, color="gray", linestyle="--", alpha=0.7, label="max_depth ales (5)")
plt.xlabel("Adâncime maximă (max_depth)")
plt.ylabel("Acuratețe (%)")
plt.title("Train vs Cross-Validation în funcție de adâncimea arborilor")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("grafic_overfitting.png", dpi=150)
plt.close()
print("Salvat: grafic_overfitting.png")




model = RandomForestClassifier(n_estimators=200, max_depth=5,
                               class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=["D", "L", "W"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Egal", "Victorie Oaspeți", "Victorie Gazde"])
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Greens", colorbar=True)
plt.title("Matricea de confuzie")
plt.tight_layout()
plt.savefig("grafic_confusion_matrix.png", dpi=150)
plt.close()
print("Salvat: grafic_confusion_matrix.png")



model_full = RandomForestClassifier(n_estimators=200, max_depth=5,
                                     class_weight="balanced", random_state=42)
model_full.fit(X, y)
importances = pd.Series(model_full.feature_importances_, index=all_cols).sort_values(ascending=True)

plt.figure(figsize=(9, 10))
plt.barh(importances.index, importances.values * 100, color="#0d7a3e")
plt.xlabel("Importanță (%)")
plt.title("Importanța variabilelor în model")
plt.tight_layout()
plt.savefig("grafic_feature_importance.png", dpi=150)
plt.close()
print("Salvat: grafic_feature_importance.png")


combos = {
    "Doar base": base_cols,
    "Base + formă": base_cols + ["home_form_pts_last5", "away_form_pts_last5"],
    "Base + clean sheets": base_cols + ["home_clean_sheets_rate", "away_clean_sheets_rate"],
    "Base + win streak": base_cols + ["home_win_streak", "away_win_streak"],
    "Base + peste 2.5": base_cols + ["home_over_2_5_rate", "away_over_2_5_rate"],
    "Base + BTTS": base_cols + ["home_btts_rate", "away_btts_rate"],
    "Toate features": all_cols
}

combo_names = []
combo_scores = []

for name, cols in combos.items():
    Xc = df[cols]
    m = RandomForestClassifier(n_estimators=200, max_depth=5,
                               class_weight="balanced", random_state=42)
    cv = cross_val_score(m, Xc, y, cv=5, scoring="accuracy")
    combo_names.append(name)
    combo_scores.append(cv.mean() * 100)

plt.figure(figsize=(10, 5))
bars = plt.bar(combo_names, combo_scores, color="#0d7a3e")
plt.axhline(y=33.3, color="red", linestyle="--", alpha=0.7, label="Baseline aleatoriu (33%)")
plt.ylabel("Acuratețe validare încrucișată (%)")
plt.title("Acuratețe (5-fold) în funcție de factorii incluși")
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.ylim(0, 60)
for bar, score in zip(bars, combo_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{score:.1f}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("grafic_comparatie_features.png", dpi=150)
plt.close()
print("Salvat: grafic_comparatie_features.png")

print("\nToate graficele au fost generate!")