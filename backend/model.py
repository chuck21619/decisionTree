import numpy as np
import xgboost as xgb

def train_model(x_player, y_player, x_deck, y_deck):

    model_player = xgb.XGBClassifier(n_estimators=25, max_depth=2)
    model_player.fit(x_player, y_player)

    model_deck = xgb.XGBClassifier(n_estimators=25, max_depth=2)
    model_deck.fit(x_deck, y_deck)

    player_predictions = model_player.predict(x_player)
    deck_predictions = model_deck.predict(x_deck)

    # Stack the predictions as features for the combined model
    combined_features = np.vstack((player_predictions, deck_predictions)).T

    # Train a meta-model on the combined features
    meta_model = xgb.XGBClassifier(n_estimators=25, max_depth=2)
    meta_model.fit(combined_features, y_player)  # Use player-target as meta-model's target

    return meta_model, combined_features
