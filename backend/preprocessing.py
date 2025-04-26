import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    
    df = df.fillna('none')

    player_names = sorted(df.columns.drop("winner"))

    deck_values = df.drop(columns="winner").values.ravel()
    deck_names = pd.unique(deck_values)
    deck_names.sort()

    le_players = LabelEncoder()
    le_decks = LabelEncoder()
    le_players.fit(player_names + ['none'])  # Add 'none' to handle when a player is absent
    le_decks.fit(deck_names)

    df_players = df.drop(columns="winner")
    deck_array = df_players.values

    player_array = np.tile(df_players.columns.to_numpy(), (deck_array.shape[0], 1))
    encoded_player_array = le_players.transform(player_array.ravel()).reshape(player_array.shape)
    none_player_code = le_players.transform(["none"])[0]
    encoded_players = np.where(deck_array != "none", encoded_player_array, none_player_code)

    encoded_decks = le_decks.transform(deck_array.ravel()).reshape(deck_array.shape)
    encoded_winner = le_players.transform(df["winner"])

    return encoded_players, encoded_decks, encoded_winner, le_players, le_decks
