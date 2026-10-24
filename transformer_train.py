
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from transformer_model import TransformerModel


class Trainer:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        self.sequence_length = 3
        self.number_of_epochs = 100
        self.learning_rate = 0.001
        self.batch_size = 64
        self.test_size = 0.2
        self.random_seed = 42

        self.songs = []
        self.training_songs = []
        self.testing_songs = []

        self.chord_to_id = {}
        self.training_examples = []
        self.testing_examples = []

        self.model = None

        self.history = {
            "train_loss": [],
            "test_loss": [],
            "accuracy": [],
            "precision": [],
            "f1_score": []
        }

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def load_dataset(self):
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            self.songs = json.load(file)

        if len(self.songs) < 2:
            raise ValueError(
                "At least two songs are required for a train/test split."
            )

        print("Songs loaded:", len(self.songs), flush=True)

    def split_dataset(self):
        self.training_songs, self.testing_songs = train_test_split(
            self.songs,
            test_size=self.test_size,
            random_state=self.random_seed,
            shuffle=True
        )

        print("Training songs:", len(self.training_songs))
        print("Testing songs:", len(self.testing_songs))

    def build_vocabulary(self):
        chord_set = set()

        # Use the same vocabulary for training and testing.
        # This ensures that every chord has a consistent ID.
        for song in self.songs:
            for chord in song["chords"]:
                chord_set.add(chord)

        sorted_chords = sorted(chord_set)

        self.chord_to_id = {
            chord: chord_id
            for chord_id, chord in enumerate(sorted_chords)
        }

        print("Unique chords:", len(self.chord_to_id))

    def create_examples(self, songs):
        examples = []

        for song in songs:
            chords = song["chords"]

            for index in range(
                self.sequence_length,
                len(chords)
            ):
                previous_chords = chords[
                    index - self.sequence_length:index
                ]

                target_chord = chords[index]

                input_ids = [
                    self.chord_to_id[chord]
                    for chord in previous_chords
                ]

                target_id = self.chord_to_id[target_chord]

                examples.append((input_ids, target_id))

        return examples

    def prepare_data(self):
        self.split_dataset()
        self.build_vocabulary()

        self.training_examples = self.create_examples(
            self.training_songs
        )

        self.testing_examples = self.create_examples(
            self.testing_songs
        )

        if not self.training_examples:
            raise ValueError(
                "No training examples. Songs need more than "
                "three chords."
            )

        if not self.testing_examples:
            raise ValueError(
                "No testing examples. Check the test songs."
            )

        print("Training examples:", len(self.training_examples))
        print("Testing examples:", len(self.testing_examples))

    def create_loader(self, examples, shuffle=False):
        X = torch.tensor(
            [example[0] for example in examples],
            dtype=torch.long
        )

        y = torch.tensor(
            [example[1] for example in examples],
            dtype=torch.long
        )

        dataset = TensorDataset(X, y)

        generator = torch.Generator()
        generator.manual_seed(self.random_seed)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None
        )

        return loader

    def evaluate(self, examples, loss_function):
        self.model.eval()

        loader = self.create_loader(examples)

        total_loss = 0.0
        total_examples = 0

        actual_chords = []
        predicted_chords = []

        with torch.no_grad():
            for input_tensor, target_tensor in loader:
                predictions = self.model(input_tensor)

                loss = loss_function(
                    predictions,
                    target_tensor
                )

                batch_count = len(target_tensor)

                total_loss += loss.item() * batch_count
                total_examples += batch_count

                predicted_ids = torch.argmax(
                    predictions,
                    dim=1
                )

                actual_chords.extend(target_tensor.tolist())
                predicted_chords.extend(predicted_ids.tolist())

        average_loss = total_loss / total_examples

        accuracy = accuracy_score(
            actual_chords,
            predicted_chords
        )

        precision = precision_score(
            actual_chords,
            predicted_chords,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            actual_chords,
            predicted_chords,
            average="macro",
            zero_division=0
        )

        return average_loss, accuracy, precision, f1

    def train(self):
        print("Starting Transformer training...", flush=True)

        number_of_chords = len(self.chord_to_id)

        self.model = TransformerModel(
            number_of_chords=number_of_chords,
            sequence_length=self.sequence_length
        )

        loss_function = nn.CrossEntropyLoss()

        # optimizer = optim.Adam(
        #     self.model.parameters(),
        #     lr=self.learning_rate
        # )
        # optimizer = optim.RMSprop(
        #         self.model.parameters(),
        #         lr=self.learning_rate)
        optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9
                )

        train_loader = self.create_loader(
            self.training_examples,
            shuffle=True
        )

        print("Batches per epoch:", len(train_loader), flush=True)

        for epoch in range(self.number_of_epochs):
            self.model.train()

            total_training_loss = 0.0
            total_examples = 0

            for batch_number, (input_tensor, target_tensor) in enumerate(
                train_loader,
                start=1
            ):
                optimizer.zero_grad()

                predictions = self.model(input_tensor)

                loss = loss_function(
                    predictions,
                    target_tensor
                )

                loss.backward()
                optimizer.step()

                batch_count = len(target_tensor)

                total_training_loss += loss.item() * batch_count
                total_examples += batch_count

                if batch_number % 100 == 0:
                    print(
                        f"Epoch {epoch + 1} | "
                        f"Batch {batch_number}/{len(train_loader)}",
                        flush=True
                    )

            training_loss = total_training_loss / total_examples

            (
                testing_loss,
                testing_accuracy,
                testing_precision,
                testing_f1
            ) = self.evaluate(
                self.testing_examples,
                loss_function
            )

            self.history["train_loss"].append(training_loss)
            self.history["test_loss"].append(testing_loss)
            self.history["accuracy"].append(testing_accuracy)
            self.history["precision"].append(testing_precision)
            self.history["f1_score"].append(testing_f1)

            print(
                f"Epoch {epoch + 1}/{self.number_of_epochs} | "
                f"Train Loss: {training_loss:.4f} | "
                f"Test Loss: {testing_loss:.4f} | "
                f"Accuracy: {testing_accuracy:.4f} | "
                f"Precision: {testing_precision:.4f} | "
                f"F1: {testing_f1:.4f}",
                flush=True
            )

    def plot_metrics(self):
        epochs = range(1, len(self.history["accuracy"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Classification metrics.
        axes[0].plot(
            epochs,
            self.history["accuracy"],
            label="Accuracy"
        )

        axes[0].plot(
            epochs,
            self.history["precision"],
            label="Macro Precision"
        )

        axes[0].plot(
            epochs,
            self.history["f1_score"],
            label="Macro F1-score"
        )

        axes[0].set_title("Transformer Classification Metrics")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(True)

        # Training and testing loss.
        axes[1].plot(
            epochs,
            self.history["train_loss"],
            label="Training Loss"
        )

        axes[1].plot(
            epochs,
            self.history["test_loss"],
            label="Testing Loss"
        )

        axes[1].set_title("Transformer Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Cross-Entropy Loss")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("transformer_training_metrics.png", dpi=300)

        print(
            "Metrics graph saved to transformer_training_metrics.png"
        )

        plt.show()

    def save_model(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "chord_to_id": self.chord_to_id,
            "sequence_length": self.sequence_length,
            "random_seed": self.random_seed,
            "embedding_dim": 64,
            "number_of_heads": 4,
            "number_of_layers": 2
        }

        model_path = "transformer_chords_sgd.pth"

        torch.save(checkpoint, model_path)

        print("Model saved to:", model_path)

    def run(self):
        self.load_dataset()
        self.prepare_data()
        self.train()
        self.save_model()
        self.plot_metrics()


if __name__ == "__main__":
   
    dataset_path =  "songs.json"
    trainer = Trainer(dataset_path)
    trainer.run()
