import os
import pandas as pd

TEAMS_FOLDER = r"C:\licenta\football-predictor\backend\teams"

max_matches = 0
max_team = ""
nrechipe=0
eliminat=0

for f in os.listdir(TEAMS_FOLDER):
    if f.endswith('.csv') and f != 'teams_index.csv':
        df = pd.read_csv(os.path.join(TEAMS_FOLDER, f))
        name = ' '.join(f.replace('.csv', '').split('_')[1:])
        nrechipe = nrechipe+1
        if len(df) > max_matches:
            max_matches = len(df)
            max_team = name
        if len(df)<= 35:
            eliminat= eliminat+1
        print(f"{name}: {len(df)} meciuri")

print(f"\nMaxim: {max_team} cu {max_matches} meciuri")

def compute_confidence(probs, num_matches):
    model_conf = max(probs)
    if num_matches >= 100:
        data_quality = 1.0
    elif num_matches >= 70:
        data_quality = 0.7
    elif num_matches >= 38:
        data_quality = 0.4
    else:
        data_quality = 0.1
    final_score = (model_conf * 0.6) + (data_quality * 0.4)
    if final_score >= 0.60:
        label = "Ridicată"
    elif final_score >= 0.50:
        label = "Medie"
    else:
        label = "Scăzută"
    return round(final_score * 100, 1), label

# Simulare scenarii
print("Scenariu maxim posibil (100+ meciuri, model_conf=1.0):")
print(compute_confidence([1.0, 0.0, 0.0], 114))

print("Scenariu realist bun (100+ meciuri, model_conf=0.65):")
print(compute_confidence([0.65, 0.20, 0.15], 114))

print("Scenariu realist mediu (38 meciuri, model_conf=0.55):")
print(compute_confidence([0.55, 0.25, 0.20], 38))

print("Scenariu slab (2 meciuri, model_conf=0.40):")
print(compute_confidence([0.40, 0.35, 0.25], 2))

print(nrechipe)
print(eliminat)