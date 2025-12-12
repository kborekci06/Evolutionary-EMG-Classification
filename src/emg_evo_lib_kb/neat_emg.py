#%% Imports
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import neat
import graphviz
import random

from emg_evo_lib_kb import build_feature_dataset, plot_confusion_matrix
# %% Load data and split for NEAT

def load_data_and_split_neat(root, emg_column_names, valid_classes,
                             test_size = 0.2, val_size = 0.1, 
                             random_state = 42):
    """
    Load EMG features and labels (classes 1-6),
    then split into train, val, and test sets.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X, y, meta = build_feature_dataset(root, emg_column_names, valid_classes, verbose=True)

    print("\nUnique labels in y:", np.unique(y))
    print("Total samples:", len(y))

    # First: train + temp (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Second: temp -> val + test (we can skip stratify here to avoid tiny class issues)
    val_ratio = val_size / (1.0 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1.0 - val_ratio,
        stratify=None,
        random_state=random_state,
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val:   {X_val.shape[0]}")
    print(f"  Test:  {X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test

#%% NEAT Evaluation Helper Functions

def evaluate_genome(genome, config, X, y, valid_classes):
    """
    Compute classification accuracy of a single genome on (X, y).

    Assumes:
        - X: shape (N_samples, 64)
        - y: labels in {1,2,3,4,5,6}
        - Genome corresponds to a FeedForwardNetwork with 64 inputs, 6 outputs.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    correct = 0
    total = len(y)

    for xi, yi in zip(X, y):
        output = net.activate(xi.tolist())   # 6 raw outputs
        pred_idx = int(np.argmax(output))   # 0..5
        pred_label = valid_classes[pred_idx]  # map index -> class label (1..6)
        if pred_label == yi:
            correct += 1

    return correct / total if total > 0 else 0.0


def eval_genomes(genomes, config, X_train, y_train, valid_classes):
    """
    NEAT callback: evaluate all genomes in the current population.
    Sets genome.fitness for each genome.
    """
    for genome_id, genome in genomes:
        acc = evaluate_genome(genome, config, X_train, y_train, valid_classes)
        genome.fitness = acc

#%% Helper Function to plot the fitness over generations
def plot_fitness_stats(stats):
    """
    Plot best and average fitness over generations.
    """
    best_fitness = stats.get_fitness_stat(max)          # best per generation
    avg_fitness  = stats.get_fitness_stat(np.mean)      # average per generation

    generations  = range(len(best_fitness))

    plt.figure(figsize=(8, 5))
    plt.plot(generations, best_fitness, label="Best Fitness")
    plt.plot(generations, avg_fitness, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Accuracy)")
    plt.title("NEAT Fitness Over Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#%% Function to save NEAT genome
def draw_genome_graphviz(config, genome, filename: str, node_names=None, view=False):
    """
    Save a NEAT genome topology diagram to <filename>.png using graphviz.
    """
    if node_names is None:
        node_names = {}

    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR")

    # Input nodes (neat-python uses negative keys: -1, -2, ... -num_inputs)
    for i in range(config.genome_config.num_inputs):
        node_id = -i - 1
        label = node_names.get(node_id, f"in{i}")
        dot.node(str(node_id), label, shape="box", style="filled", fillcolor="#97C2FC")

    # Output nodes (usually 0..num_outputs-1)
    for o in range(config.genome_config.num_outputs):
        label = node_names.get(o, f"out{o}")
        dot.node(str(o), label, shape="box", style="filled", fillcolor="#FFB07C")

    # Hidden nodes (everything else)
    hidden_ids = [nid for nid in genome.nodes.keys()
                  if nid not in range(config.genome_config.num_outputs)]
    for nid in hidden_ids:
        dot.node(str(nid), f"h{nid}", style="filled", fillcolor="#C4F0C2")

    # Connections
    for cg in genome.connections.values():
        src, dst = cg.key
        if cg.enabled:
            dot.edge(str(src), str(dst), color="black", label=f"{cg.weight:.2f}")
        else:
            # Optional: show disabled edges as dotted gray (can be noisy)
            # dot.edge(str(src), str(dst), color="gray", style="dotted")
            pass

    dot.render(filename, view=view)
    return filename + ".png"


#%% Main NEAT Pipeline for EMG Classification
def run_neat_emg(root, emg_column_names, valid_classes,
                 neat_config_path, n_generations = 50):
    """
    Full NEAT training + evaluation pipeline:

    - Load and split EMG dataset
    - Scale features (StandardScaler)
    - Run NEAT for n_generations on training data
    - Evaluate best genome on val and test sets
    - Print metrics and confusion matrix for test set
    """
    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_and_split_neat(root, emg_column_names, valid_classes)

    # 2. Scale features: fit on train, apply to val/test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 3. Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(neat_config_path),
    )

    # 4. Create population and reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # GEN-0: pick a random individual from the initial population and draw it
    gen0_genome_id = random.choice(list(population.population.keys()))
    gen0_genome = population.population[gen0_genome_id]

    output_labels = {
        0: "Class1", 1: "Class2", 2: "Class3", 3: "Class4", 4: "Class5", 5: "Class6"
    }
    draw_genome_graphviz(
        config,
        gen0_genome,
        filename="fig_gen0_random",
        node_names=output_labels,
        view=False
    )
    print(f"Saved gen-0 random topology diagram: fig_gen0_random.png")

    # 5. Run NEAT
    print("\nRunning NEAT for", n_generations, "generations...")
    winner = population.run(
        lambda genomes, cfg: eval_genomes(genomes, cfg, X_train_scaled, y_train, valid_classes),
        n_generations,
    )

    print("\n=== NEAT Evolution Complete ===")
    print("Best genome:\n", winner)

    # Draw winner genome
    draw_genome_graphviz(
    config,
    winner,
    filename="fig_neat_winner",
    node_names=output_labels,
    view=False
    )
    print("Saved winner topology diagram: fig_neat_winner.png")

    # 6. Evaluate winner on train, val, test
    print("\n--- Winner Performance ---")

    train_acc = evaluate_genome(winner, config, X_train_scaled, y_train, valid_classes)
    val_acc   = evaluate_genome(winner, config, X_val_scaled, y_val, valid_classes)
    test_acc  = evaluate_genome(winner, config, X_test_scaled, y_test, valid_classes)

    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Val Accuracy:   {val_acc * 100:.2f}%")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%\n")

    # 7. Detailed classification report on test set
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    y_test_pred = []
    for xi in X_test_scaled:
        output = net.activate(xi.tolist())
        pred_idx = int(np.argmax(output))
        pred_label = valid_classes[pred_idx]
        y_test_pred.append(pred_label)

    y_test_pred = np.array(y_test_pred)

    print("Classification report (NEAT, Test):")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=valid_classes,
            digits=3,
        )
    )

    cm = confusion_matrix(y_test, y_test_pred, labels=valid_classes)
    plot_confusion_matrix(
        cm,
        class_labels=valid_classes,
        title="NEAT Confusion Matrix (Test Set)",
    )

    # 8. Plot fitness over generations
    plot_fitness_stats(stats)

    return winner