import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_model(X_players, X_decks, y_encoded, le_players, le_decks):
    # Split data (no stratify to avoid issues with small classes)
    X_players_train, X_players_test, X_decks_train, X_decks_test, y_train, y_test = train_test_split(
        X_players, X_decks, y_encoded, test_size=0.2, random_state=42
    )

    # Player model
    player_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le_players.classes_),  # Total number of players, even non-winners
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.6,
        colsample_bytree=0.6,
        early_stopping_rounds=3
    )
    player_model.fit(X_players_train, y_train, eval_set=[(X_players_test, y_test)])

    # Deck model
    deck_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le_decks.classes_),
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.6,
        colsample_bytree=0.6,
        early_stopping_rounds=3
    )
    deck_model.fit(X_decks_train, y_train, eval_set=[(X_decks_test, y_test)])

    # Meta-model (example here as a simple average of predictions)
    meta_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le_players.classes_),
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.6,
        colsample_bytree=0.6,
        early_stopping_rounds=3
    )

    # Using both models for meta-model predictions
    player_preds = player_model.predict(X_players_test)
    deck_preds = deck_model.predict(X_decks_test)

    # Combine predictions
    meta_X = pd.DataFrame({'player_preds': player_preds, 'deck_preds': deck_preds})
    meta_model.fit(meta_X, y_test)

    # Evaluate accuracy (simplified for now)
    accuracy = (meta_model.score(meta_X, y_test)) * 100

    return player_model, deck_model, meta_model, accuracy, X_players
