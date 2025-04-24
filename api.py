from fastapi import FastAPI, Request
from pydantic import BaseModel
from prediction import predict_winner
from preprocessing import encode_data
from model import train_model
from data_generation import generate_dataset

# Step 1: Load + train model on startup
df = generate_dataset()
df, player_columns, le_players, le_decks = encode_data(df)
model, _, X = train_model(df, player_columns, le_players)

# Step 2: Set up FastAPI
app = FastAPI()

# Step 3: Define input format
class GameInput(BaseModel):
    players: dict

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
