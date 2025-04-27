import numpy as np
import xgboost as xgb
from backend.preprocessing import encode_game_input

# Variables to be used in the entire file scope
model_player = None
model_deck = None
model_meta = None
le_input_players = None
le_target_players = None
le_input_decks = None
le_target_decks = None


def post_filter_prediction(pred_probs, allowed_labels, label_encoder):
    """
    pred_probs: array of shape (num_classes,)
    allowed_labels: set of label names (e.g., {'deckA', 'deckB'})
    label_encoder: the LabelEncoder used to map classes
    """
    classes = label_encoder.inverse_transform(np.arange(pred_probs.shape[0]))
    
    for idx, label in enumerate(classes):
        if label not in allowed_labels:
            pred_probs[idx] = 0
    
    # Renormalize (optional but good practice)
    if pred_probs.sum() > 0:
        pred_probs = pred_probs / pred_probs.sum()
    
    return np.argmax(pred_probs)


def train_model(x_player, y_player, le_input_players_input, le_target_players_input, x_deck, y_deck, le_input_decks_input, le_target_decks_input):
    """
    Train player and deck models and create a meta-model.
    """
    global model_meta, model_player, model_deck, le_input_players, le_target_players, le_input_decks, le_target_decks

    # Save the label encoders to file scope
    le_input_players = le_input_players_input
    le_target_players = le_target_players_input
    le_input_decks = le_input_decks_input
    le_target_decks = le_target_decks_input

    # Train player model
    model_player = xgb.XGBClassifier(n_estimators=50, max_depth=3)
    model_player.fit(x_player, y_player)

    # Train deck model
    model_deck = xgb.XGBClassifier(n_estimators=50, max_depth=3)
    model_deck.fit(x_deck, y_deck)

    # Post-filter deck predictions (only keep decks that were in the game)
    deck_probs = model_deck.predict_proba(x_deck)  # Get deck prediction probabilities
    deck_filtered_preds = np.zeros(deck_probs.shape[0], dtype=int)

    # Iterate over each game (row)
    for i in range(x_deck.shape[0]):
        allowed_decks = set(le_input_decks.inverse_transform(x_deck[i]))  # get decks used in the game
        deck_filtered_preds[i] = post_filter_prediction(deck_probs[i], allowed_decks, le_target_decks)

    # Post-filter player predictions (only keep players that were in the game)
    player_probs = model_player.predict_proba(x_player)  # Get player prediction probabilities
    player_filtered_preds = np.zeros(player_probs.shape[0], dtype=int)

    # Iterate over each game (row)
    for i in range(x_player.shape[0]):
        allowed_players = set(le_input_players.inverse_transform(x_player[i]))  # get players from the current game row
        player_filtered_preds[i] = post_filter_prediction(player_probs[i], allowed_players, le_target_players)

    # Stack the predictions as features for the combined model
    combined_features = np.vstack((player_filtered_preds, deck_filtered_preds)).T
    print(f"combined_features training:{combined_features}")
    # Train a meta-model on the combined features
    model_meta = xgb.XGBClassifier(n_estimators=50, max_depth=3)
    model_meta.fit(combined_features, y_player)  # Use player-target as meta-model's target

    return model_meta, combined_features


def model_predict(game_input):
    """
    Make predictions based on game input (player-deck pairs).
    """
    # Step 1: Encode the game_input using the provided encoders
    encoded_game_input = encode_game_input(game_input, le_input_players, le_input_decks)
    print(f"encoded_game_input:{encoded_game_input}")

    # Step 2: Generate player and deck predictions
    x_player_input = encoded_game_input[:, 0].reshape(1, -1)  # Player features
    print(f"x_player_input:{x_player_input}")
    x_deck_input = encoded_game_input[:, 1].reshape(1, -1)    # Deck features
    print(f"x_deck_input:{x_deck_input}")

    # Predict probabilities for player and deck models
    player_probs = model_player.predict_proba(x_player_input)
    deck_probs = model_deck.predict_proba(x_deck_input)

    # Step 3: Post-filter the predictions (only allow the decks/players used in the game)
    allowed_players = set(game_input.dict().keys())  # Get the players from the game input (as a dictionary)
    allowed_decks = set(game_input.dict().values())  # Get the decks from the game input (as a dictionary)

    # Post-filter predictions for players
    player_filtered_pred = post_filter_prediction(player_probs[0], allowed_players, le_target_players)

    # Post-filter predictions for decks
    deck_filtered_pred = post_filter_prediction(deck_probs[0], allowed_decks, le_target_decks)

    # Step 4: Combine the predictions and predict the winner using the meta-model
    # Combine the player and deck filtered predictions into a single feature vector
    combined_features = np.array([player_filtered_pred, deck_filtered_pred]).reshape(1, -1)
    print(f"combined_features prediction:{combined_features}")

    # Use the meta-model to predict the final winner
    final_prediction = model_meta.predict(combined_features)  # Using player model for the final prediction

    # Return the final predicted winner
    predicted_winner = le_target_players.inverse_transform(final_prediction)
    return predicted_winner[0]
