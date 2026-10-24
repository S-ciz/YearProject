
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

# Plotting
import matplotlib.pyplot as plt

from lstm_model import LSTMModel


class Trainer:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        self.sequence_length = 3
        self.number_of_epochs = 100
        self.learning_rate = 0.001
        self.test_size = 0.2
        self.random_seed = 42

        self.songs = []
        self.chord_to_id = {}
        self.training_examples = []
        self.testing_examples = []

        self.model = None

        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        self.history = {
            "train_loss": [],
            "test_loss": [],
            "accuracy": [],
            "precision": [],
            "f1_score": [] }

    def load_dataset(self):
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            self.songs = json.load(file)

        if len(self.songs) < 2:
            raise ValueError(
                "At least two songs are required for a train/test split."
            )

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

        for song in self.songs:
            for chord in song["chords"]:
                chord_set.add(chord)

        sorted_chords = sorted(chord_set)

        for chord_id, chord in enumerate(sorted_chords):
            self.chord_to_id[chord] = chord_id

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

                input_ids = []

                for chord in previous_chords:
                    input_ids.append(self.chord_to_id[chord])

                target_id = self.chord_to_id[target_chord]

                example = (input_ids, target_id)
                examples.append(example)

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

        if len(self.training_examples) == 0:
            raise ValueError(
                "No training examples. Songs need more than "
                "three chords."
            )

        if len(self.testing_examples) == 0:
            raise ValueError(
                "No testing examples. Check the test songs."
            )

        print("Training examples:", len(self.training_examples))
        print("Testing examples:", len(self.testing_examples))

    def evaluate(self, examples, loss_function):
        self.model.eval()

        total_loss = 0.0
        actual_chords = []
        predicted_chords = []

        with torch.no_grad():
            for input_ids, target_id in examples:
                input_tensor = torch.tensor(
                    [input_ids],
                    dtype=torch.long
                )

                target_tensor = torch.tensor(
                    [target_id],
                    dtype=torch.long
                )

                predictions = self.model(input_tensor)

                loss = loss_function(
                    predictions,
                    target_tensor
                )

                total_loss = total_loss + loss.item()

                predicted_id = torch.argmax(
                    predictions,
                    dim=1
                ).item()

                actual_chords.append(target_id)
                predicted_chords.append(predicted_id)

        average_loss = total_loss / len(examples)

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
        print("Starting to train...", flush=True)

        number_of_chords = len(self.chord_to_id)

        self.model = LSTMModel(number_of_chords)

        loss_function = nn.CrossEntropyLoss()

        # optimizer = optim.Adam(
        #     self.model.parameters(),
        #     lr=self.learning_rate
        # )
        # optimizer = optim.RMSprop(
        #     self.model.parameters(),
        #     lr=self.learning_rate)
        #sgd with momentum
        optimizer = optim.SGD(
        self.model.parameters(),
        lr=self.learning_rate,
        momentum=0.9
        )


        # Convert all examples to tensors once, not every epoch.
        X_train = torch.tensor(
            [example[0] for example in self.training_examples],
            dtype=torch.long
        )

        y_train = torch.tensor(
            [example[1] for example in self.training_examples],
            dtype=torch.long
        )

        train_dataset = TensorDataset(X_train, y_train)

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True
        )

        print("Training examples:", len(train_dataset), flush=True)
        print("Batches per epoch:", len(train_loader), flush=True)

        for epoch in range(self.number_of_epochs):
            self.model.train()

            total_training_loss = 0.0

            for batch_number, (input_tensor, target_tensor) in enumerate(
                train_loader, start=1
            ):
                optimizer.zero_grad()

                predictions = self.model(input_tensor)

                loss = loss_function(
                    predictions,
                    target_tensor
                )

                loss.backward()
                optimizer.step()

                total_training_loss += loss.item() * len(target_tensor)

                # Show progress every 100 batches.
                if batch_number % 100 == 0:
                    print(
                        f"Epoch {epoch + 1} | "
                        f"Batch {batch_number}/{len(train_loader)}",
                        flush=True
                    )

            training_loss = total_training_loss / len(train_dataset)

            (
                testing_loss,
                testing_accuracy,
                testing_precision,
                testing_f1
            ) = self.evaluate(
                self.testing_examples,
                loss_function
            )
            # for plotting...
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


    def save_model(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "chord_to_id": self.chord_to_id,
            "sequence_length": self.sequence_length,
            "random_seed": self.random_seed
        }

    

        model_path =  "lstm_chords_sgd.pth"

        torch.save(checkpoint, model_path)

        print("Model saved to:", model_path)

    def run(self):
        self.load_dataset()
        self.prepare_data()
        self.train()
        self.save_model()
        self.plot_metrics()
    
    def plot_metrics(self):
        epochs = range(1, len(self.history["accuracy"]) + 1)

        # Create two plots side by side.
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot accuracy, precision, and F1-score.
        axes[0].plot(
            epochs,
            self.history["accuracy"],
            label="Accuracy"
        )

        axes[0].plot(
            epochs,
            self.history["precision"],
            label="Precision"
        )

        axes[0].plot(
            epochs,
            self.history["f1_score"],
            label="F1-score"
        )

        axes[0].set_title("LSTM Classification Metrics")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(True)

        # Plot training and testing loss.
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

        axes[1].set_title("LSTM Training and Testing Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        # Save the graph for your research report.
        plt.savefig("lstm_rms_metrics.png", dpi=300)

        print("Metrics graph saved to lstm_training_metrics.png")

        plt.show()



if __name__ == "__main__":
   
    trainer = Trainer(str("songs.json"))
    trainer.run()
