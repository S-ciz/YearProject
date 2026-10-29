const express = require("express");
const RNN = require("./rnn");
const fs = require("fs");
const path = require("path");
const cors = require("cors")

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors())
app.use(express.json());

// Load the saved model once when the server starts.
const MODEL_PATH = path.join(
  __dirname,
  "rnn_chord_model.json"
);

if (!fs.existsSync(MODEL_PATH)) {
  throw new Error(
    `Saved model not found: ${MODEL_PATH}. Train and save the model first.`
  );
}

const savedModel = JSON.parse(
  fs.readFileSync(MODEL_PATH, "utf8")
);

// Restore the model configuration and chord vocabulary.
const model = new RNN({
  ...savedModel.config,
  chordVocabulary: savedModel.chordVocabulary
});

// Restore trained weights and biases.
model.Wxh = savedModel.weights.Wxh;
model.Whh = savedModel.weights.Whh;
model.Why = savedModel.weights.Why;
model.bh = savedModel.weights.bh;
model.by = savedModel.weights.by;

console.log("Trained RNN model loaded successfully.");

// Generate multiple chords using the trained model.
function generateChords(inputChords, numChords) {
  const generatedChords = [];

  // Keep extending the sequence with each prediction.
  const sequence = [...inputChords];

  for (let i = 0; i < numChords; i++) {
    const prediction = model.predictWithSequence(sequence);

    if (!prediction || !prediction.chord) {
      throw new Error("The model failed to predict the next chord.");
    }

    generatedChords.push(prediction.chord);

    // Add the predicted chord to the context for the next prediction.
    sequence.push(prediction.chord);
  }

  return generatedChords;
}

// POST /generate
app.post("/generate", (req, res) => {
  try {
    const { chords, num_chords } = req.body;

    // Validate the input sequence.
    if (
      !Array.isArray(chords) ||
      chords.length === 0 ||
      !chords.every(
        chord =>
          typeof chord === "string" &&
          chord.trim().length > 0
      )
    ) {
      return res.status(400).json({
        error: "chords must be a non-empty array of chord strings."
      });
    }

    // Validate the requested number of chords.
    if (
      !Number.isInteger(num_chords) ||
      num_chords < 1 ||
      num_chords > 100
    ) {
      return res.status(400).json({
        error: "num_chords must be an integer between 1 and 100."
      });
    }

    // Ensure all input chords exist in the trained vocabulary.
    const vocabulary = new Set(savedModel.chordVocabulary);

    const unknownChords = chords.filter(
      chord => !vocabulary.has(chord)
    );

    if (unknownChords.length > 0) {
      return res.status(400).json({
        error: "Some input chords are not in the model vocabulary.",
        unknown_chords: [...new Set(unknownChords)]
      });
    }

    const generatedChords = generateChords(
      chords,
      num_chords
    );

    return res.json({
      generated_chords: generatedChords
    });
  } catch (error) {
    console.error("Chord generation error:", error);

    return res.status(500).json({
      error: "Failed to generate chords."
    });
  }
});

// Optional health-check endpoint.
app.get("/", (req, res) => {
  res.json({
    message: "Chord RNN API is running.",
    model_loaded: true,
    endpoint: "POST /generate"
  });
});

app.listen(PORT, "127.0.0.1", () => {
  console.log(`Server running at http://127.0.0.1:${PORT}`);
});