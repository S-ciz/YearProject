
import torch
import torch.nn as nn


class TransformerModel(nn.Module):

    def __init__(
        self,
        number_of_chords,
        sequence_length=3,
        embedding_dim=64,
        number_of_heads=4,
        number_of_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=number_of_chords,
            embedding_dim=embedding_dim
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=sequence_length,
            embedding_dim=embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=number_of_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=number_of_layers
        )

        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(
            embedding_dim,
            number_of_chords
        )

    def forward(self, input_ids):
        # input_ids shape: (batch_size, sequence_length)

        batch_size, sequence_length = input_ids.shape

        # Convert chord IDs into vectors.
        chord_embeddings = self.embedding(input_ids)

        # Create positional indices: 0, 1, 2, ...
        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        positional_embeddings = self.position_embedding(positions)

        # Combine chord and positional embeddings.
        x = chord_embeddings + positional_embeddings

        # Learn relationships between the input chords.
        x = self.transformer_encoder(x)

        # Use the final position to predict the next chord.
        last_position = x[:, -1, :]

        last_position = self.dropout(last_position)

        # Output one score per possible chord.
        predictions = self.output_layer(last_position)

        return predictions
