// main.js
document.addEventListener("DOMContentLoaded", async () => {
    try {
      const response = await fetch("https://decisiontree-api-p1eo.onrender.com/options");
      const data = await response.json();
  
      const playerIds = ["player1", "player2", "player3", "player4"];
      const deckIds = ["deck1", "deck2", "deck3", "deck4"];
  
      playerIds.forEach(id => populateDropdown(id, data.players));
      deckIds.forEach(id => populateDropdown(id, data.decks));
  
      // Show prediction section once everything is loaded
      document.getElementById("predictionSection").style.display = "block";
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
  