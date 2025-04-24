from fastapi import FastAPI, Request
from pydantic import BaseModel
from backend.prediction import predict_winner
from backend.preprocessing import encode_data
from backend.model import train_model
from backend.data_generation import generate_dataset
from backend.data_generation import get_unique_players_and_decks

# Step 1: Load + train model on startup
df = generate_dataset()
df, player_columns, le_players, le_decks = encode_data(df)
model, _, X = train_model(df, player_columns, le_players)

# Step 2: Set up FastAPI
app = FastAPI()

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
    result = predict_winner(
        game_input.players,
        model,
        X.columns.tolist(),
        le_decks,
        le_players
    )
    return {"winner": result}

