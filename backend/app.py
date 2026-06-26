from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))


from dynamic_trainer import get_pie_chart_prediction

app = Flask(__name__)
CORS(app)


TEAMS_FOLDER = r"C:\licenta\football-predictor\backend\teams"



@app.route('/teams', methods=['GET'])
def get_teams():
    try:
        folder_path = os.path.join(os.path.dirname(__file__), 'teams')

        if not os.path.exists(folder_path):
            return jsonify({"error": f"Folderul nu a fost gasit la: {folder_path}"}), 404

        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        MIN_MATCHES = 35

        teams = []
        for f in files:
            team_df = pd.read_csv(os.path.join(folder_path, f))
            if len(team_df) >= MIN_MATCHES:
                name = f.replace('.csv', '')
                if '_' in name:
                    name = ' '.join(name.split('_')[1:])
                teams.append(name)

        return jsonify(sorted(teams))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    home_team = data.get('home_team')
    away_team = data.get('away_team')
    selected_optionals = data.get('options', [])

    print(f"Predicție solicitată pentru: {home_team} vs {away_team}")


    result = get_pie_chart_prediction(home_team, away_team, selected_optionals)

    return jsonify(result)


if __name__ == '__main__':

    app.run(debug=True, port=5000)