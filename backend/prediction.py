import pandas as pd

def predict_winner(game, model, X_columns, le_decks, le_players):
    input_data = {col: game.get(col, 'none') for col in X_columns}
    df_game = pd.DataFrame([input_data])

    for col in df_game.columns:
        df_game[col] = le_decks.transform(df_game[col])

    pred = model.predict(df_game)
    return le_players.inverse_transform(pred)[0]

def predict_probabilities(game, model, X_columns, le_decks, le_players):
    input_data = {col: game.get(col, 'none') for col in X_columns}
    df_game = pd.DataFrame([input_data])

    for col in df_game.columns:
        df_game[col] = le_decks.transform(df_game[col])

    probs = model.predict_proba(df_game)
    
    # Convert numpy floats to native Python float
    return {player: float(probs[0][i]) for i, player in enumerate(le_players.classes_)}
