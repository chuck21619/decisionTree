from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from pydantic import BaseModel
from backend.prediction import predict_winner, predict_probabilities
from backend.preprocessing import encode_data
from backend.model import train_model
from backend.data_generation import generate_dataset, get_unique_players_and_decks
from sklearn.metrics import accuracy_score

# Step 1: Generate and encode data
df = generate_dataset()

# Call the function on your dataset
x_player, y_player, x_deck, y_deck, le_target_players = encode_data(df)
model, combined_features = train_model(x_player, y_player, x_deck, y_deck)

meta_predictions = model.predict(combined_features)
accuracy = accuracy_score(y_player, meta_predictions)
print(f"Meta-model accuracy: {accuracy}")
meta_player_name_predictions = le_target_players.inverse_transform(meta_predictions)
print("Meta-model player predictions:", meta_player_name_predictions)

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
class GameInput(BaseModel):
    players: dict

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
    raw_probabilities = predict_probabilities(
        game_input.players,
        model,
        X.columns.tolist(),
        le_decks,
        le_players
    )

    input_players = game_input.players.keys()

    # Build the filtered probabilities dict
    final_probabilities = {
        player: raw_probabilities.get(player, 0.0)
        for player in input_players
    }

    # Sort by probability descending
    sorted_probabilities = dict(
        sorted(final_probabilities.items(), key=lambda x: x[1], reverse=True)
    )

    # Winner is the top player
    winner = next(iter(sorted_probabilities))

    return {
        "winner": winner,
        "probabilities": sorted_probabilities
    }


