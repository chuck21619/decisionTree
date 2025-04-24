import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import random
import pandas as pd

# Sample player and deck names
player_names = ['chuck', 'pk', 'dustin', 'brittany', 'jonathan', 'austin', 'adnaan']
deck_names = [
    'windgrace', 'wyleth', 'lita', 'fortell', 'anowon', 'tenth doctor', 'kranston',
    'slip', 'estrid', 'the first sliver', 'szarekh', 'anje', 'boomerang', 'tayam',
    'otrimi', 'vhati', 'magus lucea kane', 'hazel', 'ygra', 'helga', 'hansk', 
    'naban', 'riku of many paths', 'don andres', 'Aminatou, the Fateshifter',
    'hare apparent', 'zimone', 'black panther', 'evereth', 'aesi', 'pantlaza',
    'plagon', 'oloro', 'mr. foxglove', 'giada', 'abzan tokens', 'shroofus'
]

# Function to generate one fake game record (3 or 4 players)
def generate_game():
    players = random.sample(player_names, k=random.choice([3, 4]))
    game = {player: random.choice(deck_names) for player in players}
    winner = random.choice(players)
    game['winner'] = winner
    return game

# Generate 200 fake game records
fake_games = [generate_game() for _ in range(200)]

# Convert to DataFrame
fake_df = pd.DataFrame(fake_games)

# View sample
print(fake_df.head())


# Step 1: Create a DataFrame
df = fake_df

le_players = LabelEncoder()
le_decks = LabelEncoder()

# Collect all decks across all player columns
player_columns = [col for col in df.columns if col != 'winner']
all_decks = df[player_columns].values.flatten().tolist()
all_decks.append('none')  # add a placeholder in case it's needed later
le_decks.fit(all_decks)

# Encode all player deck columns
for col in player_columns:
    df[col] = le_decks.transform(df[col])

# Encode the winners (player names)
df['winner'] = le_players.fit_transform(df['winner'])

# Step 3: Prepare the features (X) and target (y)
X = df.drop('winner', axis=1)  # Features (players' decks)
y = df['winner']  # Target (winner)

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softmax', 
    num_class=len(le_players.classes_),
    max_depth=4,           # Controls the depth of each tree to prevent overfitting
    learning_rate=0.1,     # Controls the step size of each update to prevent overfitting
    n_estimators=100,      # Number of trees
    subsample=0.8,         # Randomly sample training data to reduce overfitting
    colsample_bytree=0.8   # Randomly sample features to reduce overfitting
)
model.fit(X_train, y_train)

# Check model accuracy again
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Feature Importance (optional)
import matplotlib.pyplot as plt

# Feature Importance (if model is not overfitting)
xgb.plot_importance(model)
plt.show()










def predict_winner(game: dict):
    # Use the exact player columns used in training
    input_columns = X.columns.tolist()  # same order as training features

    # Fill in missing player slots with 'none'
    input_data = {col: game.get(col, 'none') for col in input_columns}

    # Convert to DataFrame
    df_game = pd.DataFrame([input_data])

    # Encode with deck encoder
    for col in df_game.columns:
        df_game[col] = le_decks.transform(df_game[col])

    # Predict
    pred = model.predict(df_game)
    return le_players.inverse_transform(pred)[0]


# Example usage
winner = predict_winner({
    "chuck": "kranston",
    "pk": "slip",
    "dustin": "lita"
    # the rest will default to 'none'
})
print("Predicted winner:", winner)








stub_game = {
    "chuck": "kranston",
    "pk": "slip",
    "dustin": "lita",
    "brittany": "none",  # Example: no brittany in this game, fill with 'none'
    "jonathan": "none",  # Example: no jonathan in this game
    "austin": "none",    # Example: no austin in this game
    "adnaan": "none"     # Example: no adnaan in this game
}

# Use the exact same player columns used in training
input_columns = X.columns.tolist()  # same order as training features

# Fill in missing player slots with 'none'
input_data = {col: stub_game.get(col, 'none') for col in input_columns}

# Convert to DataFrame
df_game = pd.DataFrame([input_data])

# Encode with deck encoder (same encoder as training)
for col in df_game.columns:
    df_game[col] = le_decks.transform(df_game[col])

# Predict probabilities
probs = model.predict_proba(df_game)

# Print predicted probabilities for each player
for i, player in enumerate(le_players.classes_):
    print(f"Probability of {player} winning: {probs[0][i]:.4f}")