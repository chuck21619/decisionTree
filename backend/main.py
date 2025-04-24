from data_generation import generate_dataset
from preprocessing import encode_data
from model import train_model
from prediction import predict_winner, predict_probabilities
import matplotlib.pyplot as plt
import xgboost as xgb

# Step 1: Generate and preprocess data
df = generate_dataset()
df, player_columns, le_players, le_decks = encode_data(df)

# Step 2: Train model
model, accuracy, X = train_model(df, player_columns, le_players)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 3: Plot Feature Importance
xgb.plot_importance(model)
plt.show()

# Step 4: Prediction
game_input = {
    "chuck": "kranston",
    "pk": "slip",
    "dustin": "lita"
}
print("Predicted winner:", predict_winner(game_input, model, X.columns.tolist(), le_decks, le_players))

# Optional: Probability predictions
probs = predict_probabilities(game_input, model, X.columns.tolist(), le_decks, le_players)
for player, prob in probs.items():
    print(f"Probability of {player} winning: {prob:.4f}")
