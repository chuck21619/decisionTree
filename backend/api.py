from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import RootModel
from backend.preprocessing import encode_data, encode_game_input
from backend.model import train_model, model_predict
from backend.data_generation import generate_dataset, get_unique_players_and_decks
from sklearn.metrics import accuracy_score
from typing import Dict

# Step 1: Generate and encode data
df = generate_dataset()

# Call the function on your dataset
x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks = encode_data(df)
model, combined_features = train_model(x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks)

meta_predictions = model.predict(combined_features)
accuracy = accuracy_score(y_player, meta_predictions)
meta_player_name_predictions = le_target_players.inverse_transform(meta_predictions)


for i in range(combined_features.shape[0]):
    player_pred = combined_features[i, 0]
    deck_pred = combined_features[i, 1]
    decoded_player_pred = le_target_players.inverse_transform([player_pred])[0]
    decoded_deck_pred = le_target_decks.inverse_transform([deck_pred])[0]

    input_players = le_input_players.inverse_transform(x_player[i])
    input_decks = ""
    for k in range(7):
        input_decks += " " + le_input_decks.inverse_transform([x_deck[i][k]])[0]

    true_winner = le_target_players.inverse_transform([y_player[i]])[0]

    print(f"Row {i}:")
    print(f"  Input players: {input_players}")
    print(f"  Input decks: {input_decks}")
    print(f"  Predicted Player: {decoded_player_pred}")
    print(f"  Predicted Deck: {decoded_deck_pred}")
    print(f"      Predicted Meta Player: {meta_player_name_predictions[i]}")
    print(f"      True winner: {true_winner}")
    print("-" * 40)

print(f"Meta-model accuracy: {accuracy}")

# Step 2: Set up FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with my frontend domain for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Step 3: Define input format
class GameInput(RootModel[Dict[str, str]]):
    pass

@app.get("/")
async def root():
    return {"message": "App is live"}

@app.get("/options")
async def get_options():
    players, decks = get_unique_players_and_decks()
    return {
        "players": players,
        "decks": decks
    }

@app.post("/predict")
def predict(game_input: GameInput):
    print(game_input)

    winner = model_predict(game_input)
    return {
        "winner": winner
    }
