import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def encode_targets(Y_targets):
    le_targets = LabelEncoder()
    Y_targets_encoded = le_targets.fit_transform(Y_targets)
    return Y_targets_encoded, le_targets

def train_model(X_player_train, X_deck_train, Y_targets):
    # Re-encode targets to ensure contiguous class labels starting from 0
    Y_targets_encoded, le_targets = encode_targets(Y_targets)

    # Split the data into training and test sets for validation
    X_player_train, X_player_test, Y_train, Y_test = train_test_split(X_player_train, Y_targets_encoded, test_size=0.2, random_state=42)
    X_deck_train, X_deck_test, _, _ = train_test_split(X_deck_train, Y_targets_encoded, test_size=0.2, random_state=42)
    
    # Initialize XGBoost models for both player and deck data
    player_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(Y_targets_encoded)), random_state=42, eval_metric="mlogloss", n_estimators=50, max_depth=2)
    deck_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(Y_targets_encoded)), random_state=42, eval_metric="mlogloss", n_estimators=50, max_depth=2)
    
    # Train player-focused model with early stopping
    player_model.fit(X_player_train, Y_train, 
                     eval_set=[(X_player_test, Y_test)], 
                     verbose=True)
    
    # Train deck-focused model with early stopping
    deck_model.fit(X_deck_train, Y_train, 
                   eval_set=[(X_deck_test, Y_test)], 
                   verbose=True)
    
    # Evaluate player-focused model
    player_pred = player_model.predict(X_player_test)
    player_accuracy = accuracy_score(Y_test, player_pred)
    print(f"Player-focused model accuracy (XGBoost): {player_accuracy * 100:.2f}%")
    
    # Evaluate deck-focused model
    deck_pred = deck_model.predict(X_deck_test)
    deck_accuracy = accuracy_score(Y_test, deck_pred)
    print(f"Deck-focused model accuracy (XGBoost): {deck_accuracy * 100:.2f}%")
    
    # Meta-model: Combine predictions of the player and deck models as features
    X_meta_train = np.column_stack((player_model.predict_proba(X_player_train), deck_model.predict_proba(X_deck_train)))
    X_meta_test = np.column_stack((player_model.predict_proba(X_player_test), deck_model.predict_proba(X_deck_test)))

    # Meta-model (another XGBoost)
    meta_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(Y_targets_encoded)), random_state=42, eval_metric="mlogloss", n_estimators=50, max_depth=2)
    meta_model.fit(X_meta_train, Y_train, 
                   eval_set=[(X_meta_test, Y_test)], 
                   verbose=True)
    
    # Evaluate the meta-model
    meta_pred = meta_model.predict(X_meta_test)
    meta_accuracy = accuracy_score(Y_test, meta_pred)
    print(f"Meta-model accuracy (XGBoost): {meta_accuracy * 100:.2f}%")
    
    return player_model, deck_model, meta_model
