import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    df = df.fillna('none')

    player_names = list(df.columns.drop("winner"))
    player_names.sort()
    print(f"player_names:{player_names}")
    
    deck_values = df.drop(columns="winner").values.ravel()
    deck_names = pd.unique([str(d) for d in deck_values if pd.notna(d)])
    deck_names.sort()
    print(f"deck_names:{deck_names}")
    
    le_players = LabelEncoder()
    le_decks = LabelEncoder()
    
    le_players.fit(player_names)
    le_decks.fit(deck_names)
    
    encoded_players = df.drop(columns="winner").copy()
    for col in encoded_players.columns:
        player_encoded_value = le_players.transform([col])[0]
        # Now fill: if the deck is 'none', put -1 instead of player_encoded_value
        encoded_players[col] = df[col].apply(
            lambda deck: player_encoded_value if deck != 'none' else -1
        )
        
    encoded_decks = df.drop(columns="winner").apply(lambda col: le_decks.transform(col))
    
    return encoded_players, encoded_decks, le_players, le_decks
