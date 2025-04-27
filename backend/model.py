import numpy as np
import xgboost as xgb

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
#x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks
def train_model(x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks):

    # Train player model
    model_player = xgb.XGBClassifier(n_estimators=50, max_depth=3)
    model_player.fit(x_player, y_player)

    # Train deck model
    model_deck = xgb.XGBClassifier(n_estimators=50, max_depth=3)
    model_deck.fit(x_deck, y_deck)

    # Predict player and deck probabilities
    player_predictions = model_player.predict(x_player)
    deck_predictions = model_deck.predict(x_deck)

    # Post-filter deck predictions (only keep decks that were in the game)
    allowed_decks = set(le_input_decks.inverse_transform(x_deck.ravel()))  # get decks used in the game
    deck_probs = model_deck.predict_proba(x_deck)  # Get deck prediction probabilities

    # Apply filtering for each prediction
    deck_filtered_preds = np.array([post_filter_prediction(prob, allowed_decks, le_target_decks) for prob in deck_probs])

    # Post-filter player predictions (only keep players that were in the game)
    allowed_players = set(le_input_players.inverse_transform(x_player[0]))  # get players from the game row
    player_probs = model_player.predict_proba(x_player)  # Get player prediction probabilities

    # Apply filtering for each prediction
    player_filtered_preds = np.array([post_filter_prediction(prob, allowed_players, le_target_players) for prob in player_probs])

    # Stack the predictions as features for the combined model
    combined_features = np.vstack((player_filtered_preds, deck_filtered_preds)).T

    # Train a meta-model on the combined features
    meta_model = xgb.XGBClassifier(n_estimators=50, max_depth=3)
    meta_model.fit(combined_features, y_player)  # Use player-target as meta-model's target

    return meta_model, combined_features
