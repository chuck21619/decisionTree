import pandas as pd

# Your published CSV URL from Google Sheets
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSMpIhYcYDwpGE_GlsMTClC8WaFgNGAmVa_8SH5QwloJn9aFze3ifL_XPiYJnDQtNZYWsuVZ9xUl8TF/pub?gid=0&single=true&output=csv"

def build_dataset_from_sheet(df):
    games = []
    for _, row in df.iterrows():
        game = {}
        for player in row.index:
            if player != "winner" and pd.notna(row[player]) and row[player] != '':
                game[player] = row[player]
        game["winner"] = row["winner"]
        games.append(game)
    return pd.DataFrame(games)

def generate_dataset():
    raw_data = pd.read_csv(CSV_URL)
    return build_dataset_from_sheet(raw_data)

def get_unique_players_and_decks():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRUFTglCikRQC5C1GVZal97MJBGYBP9ZDoMeYDnhtI8JRAH2Q_oZKudRWoxp7wqMCZh9GyMJhReJ3QG/pub?gid=0&single=true&output=csv"
    df = pd.read_csv(url)
    player_names = list(df.columns.drop("winner"))
    deck_names = pd.unique(df.drop(columns="winner").values.ravel())
    return sorted(player_names), sorted(deck_names)
