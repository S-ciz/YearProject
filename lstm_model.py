
import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, number_of_chords):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=number_of_chords,
            embedding_dim=32
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            in_features=64,
            out_features=number_of_chords
        )

    def forward(self, input_sequence):
        embedded_sequence = self.embedding(input_sequence)

        lstm_output, hidden_state = self.lstm(
            embedded_sequence
        )

        last_output = lstm_output[:, -1, :]

        predictions = self.output_layer(last_output)

        return predictions
