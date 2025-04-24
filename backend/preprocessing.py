from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    le_players = LabelEncoder()
    le_decks = LabelEncoder()

    player_columns = [col for col in df.columns if col != 'winner']
    all_decks = df[player_columns].values.flatten().tolist()
    all_decks.append('none')
    le_decks.fit(all_decks)

    for col in player_columns:
        df[col] = le_decks.transform(df[col])

    df['winner'] = le_players.fit_transform(df['winner'])
    return df, player_columns, le_players, le_decks
