# RL Music Emotion Management 🎵🧠

This repository contains the implementation of a Reinforcement Learning (RL) approach for EEG-based music emotion management. It is based on the framework proposed by **Dutta et al. (IEEE EMBC 2020)** and extends it with several novel deep reinforcement learning innovations to improve convergence and generalizability.

The goal of the agent is to recommend a sequence of music clips that successfully transitions a user's emotional state from an arbitrary starting emotion to a desired target emotion (e.g., "Happy") on the Geneva Emotion Wheel (GEW).

## 📁 Project Structure

* **`baseline.ipynb`**: Implementation of the paper's baseline model using a Tabular Q-Learning agent with a 9-state discretized action space and fixed reward shaping.
* **`inovation.ipynb`**: Contains four major architectural and algorithmic innovations extending the baseline model (see below).
* **`final_rl (3).pdf`**: Comprehensive project report detailing the theoretical framework, experiments, and analysis.
* **`deap-dataset/`**: Directory for the DEAP dataset metadata and rating files.
* **`outputs_baseline/`**: Generated charts, trajectories, and logs from the baseline model.
* **`outputs_innovation/`**: Generated charts, trajectories, and logs from the innovative models.

## 🚀 Innovations Implemented

The `inovation.ipynb` notebook implements the following enhancements over the baseline model:

1. **INN-1: Continuous Deep Q-Network (DQN)**
   * Replaces the 9-bucket discrete Q-table with a neural network.
   * Uses continuous 2-D valence/arousal vectors as the state representation, solving the discretization bottleneck.
2. **INN-2: Momentum-Augmented State (Momentum DQN)**
   * Expands the state space to 4-D `(v, a, Δv, Δa)` to include emotional velocity.
   * Allows the agent to perceive trajectory momentum, significantly reducing oscillation in failing subjects.
3. **INN-3: Adaptive Reward Shaping**
   * Implements a dynamic penalty `λ(t)` that grows with iteration depth (similar to curriculum learning).
   * Encourages free exploration early on while applying heavy stagnation penalties during the final critical transitions.
4. **INN-4: Multi-Target Emotion Navigation**
   * Extends the system to train and evaluate for *every* GEW emotion as a target destination.
   * Analyzes which emotional states are structurally easier (e.g., positive valence) or harder to reach.

## 📊 Dataset Requirements

This project uses the **DEAP** (Database for Emotion Analysis using Physiological Signals) dataset.
* Ensure the dataset metadata files (e.g., `participant_ratings.csv`) are correctly placed inside the `deap-dataset/Metadata/` directory.
* The data pipeline automatically applies z-score standardization per-subject followed by a `tanh` squash to normalize clip vectors.

## 💻 Running the Code

1. Install the required dependencies:
   ```bash
   pip install numpy pandas matplotlib torch stable-baselines3 tqdm
   ```
2. **Run Baseline:** Open and run all cells in `baseline.ipynb` to execute the Leave-One-Subject-Out (LOSO) cross-validation for the tabular agent. Outputs will be saved to `outputs_baseline/`.
3. **Run Innovations:** Open and run all cells in `inovation.ipynb` to train and evaluate the Continuous DQN, Momentum DQN, and Adaptive Shaping agents. Outputs and comparative analysis will be saved to `outputs_innovation/`.

## 📈 Results Overview

The Continuous DQN (INN-1) achieved a **93.8% convergence rate** (compared to the baseline's 68.8%), reducing the mean final angular error to just 5.0° across the 32 subjects. Detailed regret curves, trajectory plots, and multi-target radar charts can be found in the generated output directories.
