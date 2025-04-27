const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";

const API_BASE_URL = isLocal
    ? "http://localhost:10000"
    : "https://decisiontree-api-p1eo.onrender.com"; // Replace with your actual deployed API URL

var originalTitle = "";
var availablePlayers = [];  // Store available players

document.addEventListener("DOMContentLoaded", async () => {
    originalTitle = document.getElementById('title').innerHTML;
    try {
        document.getElementById('title').innerHTML = "Loading Players and Decks...";
        console.log(`${API_BASE_URL}/options`);
        const response = await fetch(`${API_BASE_URL}/options`);
        const data = await response.json();
        console.log(data);

        availablePlayers = data.players; // Save available players

        const playerIds = ["player1", "player2", "player3", "player4"];
        const deckIds = ["deck1", "deck2", "deck3", "deck4"];

        playerIds.forEach(id => populateDropdown(id, data.players));
        deckIds.forEach(id => populateDropdown(id, data.decks));

        // Show prediction section once everything is loaded
        document.getElementById("predictionSection").style.display = "block";
        document.getElementById('title').innerHTML = originalTitle;

    } catch (error) {
        console.error("Error loading options:", error);
    }
});

function populateDropdown(selectId, options) {
    const select = document.getElementById(selectId);
    options.forEach(option => {
        const opt = document.createElement("option");
        opt.value = option;
        opt.textContent = option;
        select.appendChild(opt);
    });
}

document.getElementById("predictButton").addEventListener("click", async () => {
    const players = {};

    // Gather the player-deck pairs from the dropdowns
    for (let i = 1; i <= 4; i++) {
        const player = document.getElementById(`player${i}`).value;
        const deck = document.getElementById(`deck${i}`).value;

        if (player && deck) {
            players[player] = deck;
        }
    }

    // Append missing players with deck 'none'
    availablePlayers.forEach(player => {
        if (!players.hasOwnProperty(player)) {
            players[player] = 'none'; // Append with 'none' deck
        }
    });

    try {
        console.log(`${API_BASE_URL}/predict`);
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(players)
        });
    
        document.getElementById('title').innerHTML = "Predicting...";
        const result = await response.json();
        console.log(result);
        document.getElementById('title').innerHTML = originalTitle;
    
        // Assuming result contains only 'winner' (a string)
        const { winner } = result;
    
        // Display the predicted winner in the alert
        alert(`Predicted winner: ${winner}`);
    
    } catch (error) {
        console.error("Prediction failed:", error);
        alert("Something went wrong with prediction.");
    }
});
