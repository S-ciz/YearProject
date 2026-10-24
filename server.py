
# server.py

import os
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS


from lstm_model import LSTMModel
from transformer_model import TransformerModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "0.0.0.0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LSTM_PATH = BASE_DIR / "models" /"lstm_chords_adam.pth"
TRANSFORMER_PATH = BASE_DIR /"models"/ "transformer_chords_adam.pth"
VOCAB_PATH = BASE_DIR / "chord_vocab.json"

# Set this to the context length used during training.
# The example LSTM project used a sequence length of 3.
SEQUENCE_LENGTH = 3

app = Flask(__name__)
CORS(app)

# Load the exact vocabulary used by your training scripts.
with open(VOCAB_PATH, "r", encoding="utf-8") as file:
    CHORD_TO_ID = json.load(file)

CHORD_TO_ID = {
    chord: int(token_id)
    for chord, token_id in CHORD_TO_ID.items()
}

ID_TO_CHORD = {
    token_id: chord
    for chord, token_id in CHORD_TO_ID.items()
}

VOCAB_SIZE = len(CHORD_TO_ID)

# Prevent random predictions from producing invalid tokens.
ALLOWED_CHORDS = set(CHORD_TO_ID.keys())

# Replace these values with actual training results if available.
METRICS = {
    "epoch": 100,
    "train_loss": 1.5548,
    "test_loss": 1.9177,
    "accuracy": 0.4171,
    "precision": 0.1417,
    "f1": 0.1168
}


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_checkpoint(model, checkpoint_path):
    """
    Supports a raw state_dict or a checkpoint dictionary
    containing 'model_state_dict' or 'state_dict'.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint_path.name}"
        )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def initialize_models():
    """
    IMPORTANT:
    Match these constructors to the architectures and
    hyperparameters used when the checkpoints were trained.
    """
    models = {}

    # Example constructors. Adjust arguments if your classes
    # require embedding_dim, hidden_dim, num_heads, etc.
    lstm = LSTMModel(vocab_size=VOCAB_SIZE)
    transformer = TransformerModel(vocab_size=VOCAB_SIZE)

    models["LSTM"] = load_checkpoint(lstm, LSTM_PATH)
    models["Transformer"] = load_checkpoint(
        transformer,
        TRANSFORMER_PATH
    )

    return models


MODELS = {}

try:
    MODELS = initialize_models()
    print(f"Models loaded successfully on {DEVICE}.")
except Exception as error:
    # Keep the server running so health checks can explain
    # a checkpoint or architecture loading problem.
    print(f"WARNING: Model loading failed: {error}")


# --------------------------------------------------
# Prediction helpers
# --------------------------------------------------

def normalize_model_output(output):
    """
    Expected output:
      [batch, sequence_length, vocab_size]
    or:
      [batch, vocab_size]

    If the model returns a tuple, the first element is used.
    Adapt this function if your model returns another structure.
    """
    if isinstance(output, (tuple, list)):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise ValueError("Model output must be a PyTorch tensor.")

    if output.ndim == 3:
        # Take the logits for the last sequence position.
        output = output[:, -1, :]
    elif output.ndim != 2:
        raise ValueError(
            f"Unexpected model output shape: {tuple(output.shape)}"
        )

    if output.shape[-1] != VOCAB_SIZE:
        raise ValueError(
            f"Model output has {output.shape[-1]} classes, "
            f"but vocabulary contains {VOCAB_SIZE} chords."
        )

    return output


def predict_next_chord(model, chord_sequence):
    """
    Predict the next chord from a list of chord names.

    Assumes the trained model takes integer token IDs
    shaped [batch_size, sequence_length].
    """
    if not chord_sequence:
        chord_sequence = ["C"]

    # Convert chords to token IDs.
    token_ids = [
        CHORD_TO_ID[chord]
        for chord in chord_sequence
        if chord in CHORD_TO_ID
    ]

    if not token_ids:
        token_ids = [CHORD_TO_ID["C"]]

    # Match the fixed context length used in training.
    token_ids = token_ids[-SEQUENCE_LENGTH:]

    # Left-pad short sequences by repeating their first token.
    while len(token_ids) < SEQUENCE_LENGTH:
        token_ids.insert(0, token_ids[0])

    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.inference_mode():
        output = model(input_tensor)
        logits = normalize_model_output(output)

        # Sample from the model distribution.
        # Use argmax instead if deterministic prediction is preferred.
        probabilities = F.softmax(logits[0], dim=-1)

        if not torch.isfinite(probabilities).all():
            raise ValueError("Model returned invalid probabilities.")

        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

    if next_token_id not in ID_TO_CHORD:
        raise ValueError(
            f"Predicted token ID {next_token_id} is not in the vocabulary."
        )

    return ID_TO_CHORD[next_token_id]


# --------------------------------------------------
# Markov baseline
# --------------------------------------------------

def generate_markov(chords, num_chords):
    """
    Small demonstration baseline, not a trained Markov model.
    Replace it with learned transition probabilities if you
    have a trained Markov implementation.
    """
    diatonic = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]

    transitions = {
        "C":   ["Am", "F", "G", "Dm"],
        "Dm":  ["G", "Am", "C", "Bdim"],
        "Em":  ["Am", "F", "C"],
        "F":   ["G", "Am", "C", "Dm"],
        "G":   ["C", "Am", "Em"],
        "Am":  ["F", "Dm", "G", "C"],
        "Bdim":["C", "Em", "G"]
    }

    sequence = [
        chord for chord in chords
        if chord in ALLOWED_CHORDS
    ]

    current = sequence[-1] if sequence else "C"

    for _ in range(num_chords):
        choices = transitions.get(current, diatonic)
        current = random.choice(choices)
        sequence.append(current)

    return sequence[-num_chords:]


# --------------------------------------------------
# API routes
# --------------------------------------------------

@app.get("/")
def home():
    return jsonify({
        "name": "CHORDLAB API",
        "status": "running",
        "device": str(DEVICE),
        "endpoints": [
            "GET /api/health",
            "POST /api/generate",
            "GET /api/analytics?model=LSTM"
        ]
    })


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "models": {
            "LSTM": "loaded" if "LSTM" in MODELS else "unavailable",
            "Transformer": (
                "loaded" if "Transformer" in MODELS else "unavailable"
            ),
            "Markov": "demo baseline"
        },
        "vocabulary_size": VOCAB_SIZE
    })


@app.post("/api/generate")
def generate():
    data = request.get_json(silent=True) or {}

    model_name = str(data.get("model", "LSTM")).strip()
    chords = data.get("chords", ["C", "Am", "F", "G"])
    key = data.get("key", "C:maj")
    tempo = data.get("tempo", 96)

    try:
        num_chords = int(data.get("num_chords", 4))
    except (TypeError, ValueError):
        return jsonify({"error": "num_chords must be an integer"}), 400

    if model_name not in {"Markov", "LSTM", "Transformer"}:
        return jsonify({
            "error": "model must be Markov, LSTM, or Transformer"
        }), 400

    if num_chords not in {2, 4, 8, 10}:
        return jsonify({
            "error": "num_chords must be 2, 4, 8, or 10"
        }), 400

    if not isinstance(chords, list):
        return jsonify({"error": "chords must be a list"}), 400

    if not all(isinstance(chord, str) for chord in chords):
        return jsonify({"error": "Each chord must be a string"}), 400

    unknown = [
        chord for chord in chords
        if chord not in ALLOWED_CHORDS
    ]

    if unknown:
        return jsonify({
            "error": "Some chords are not in the model vocabulary",
            "unknown_chords": unknown
        }), 400

    try:
        tempo = int(tempo)
        if not 40 <= tempo <= 240:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({
            "error": "tempo must be an integer from 40 to 240"
        }), 400

    try:
        if model_name == "Markov":
            generated = generate_markov(chords, num_chords)

        else:
            model = MODELS.get(model_name)

            if model is None:
                return jsonify({
                    "error": f"{model_name} model is unavailable. "
                             "Check the server logs and checkpoint."
                }), 503

            # Start with the supplied progression and repeatedly
            # predict until the requested number of new chords exists.
            context = list(chords) or ["C"]
            generated = []

            for _ in range(num_chords):
                next_chord = predict_next_chord(model, context)
                generated.append(next_chord)
                context.append(next_chord)

        return jsonify({
            "model": model_name,
            "key": key,
            "tempo": tempo,
            "input_chords": chords,
            "num_chords": num_chords,
            "chords": generated
        }), 200

    except Exception as error:
        app.logger.exception("Chord generation failed")
        return jsonify({
            "error": "Chord generation failed",
            "details": str(error)
        }), 500


@app.get("/api/analytics")
def analytics():
    model_name = request.args.get("model", "LSTM")

    if model_name not in {"Markov", "LSTM", "Transformer"}:
        return jsonify({"error": "Unknown model"}), 400

    # Replace these example values with metrics loaded from
    # the selected model's actual evaluation/training logs.
    return jsonify({
        "model": model_name,
        **METRICS
    })


if __name__ == "__main__":
    print(f"Starting CHORDLAB API on http://{HOST}:{PORT}")
    print(f"Compute device: {DEVICE}")

    app.run(
        host=HOST,
        port=PORT,
        debug=False
    )
