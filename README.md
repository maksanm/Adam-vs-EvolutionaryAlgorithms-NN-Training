# Adam-vs-EvolutionaryAlgorithms-NN-Training

This project provides an implementation and comparison of three optimizers for neural network training: Adam, Differential Evolution (DE), and the $(1+1)$ Covariance Matrix Adaptation Evolution Strategy (CMA-ES). The goal is to empirically evaluate their performance on peptide retention time prediction tasks.

## Overview

- **Optimizers:** Adam, Differential Evolution (DE), $(1+1)$-CMA-ES
- **Task:** Neural network regression on peptide datasets
- **Features:**
  - Reproducible experiment setup
  - Empirical evaluation and hyperparameter tuning
  - Visualization of learning and convergence curves, predicted vs. true values, learning dynamics

## How to Run

1. **Install dependencies**
   This project uses [Poetry](https://python-poetry.org/) for dependency management:
   ```
   poetry install
   ```

2. **Configure environment**
   Set all relevant hyperparameters and dataset paths in the `.env` file.

3. **Run training scripts**
   To train and evaluate a model with a given optimizer, use:
   ```
   poetry run python src/train_adam.py
   poetry run python src/train_de.py
   poetry run python src/train_cmaes.py
   ```

4. **Interactive experiments**
   For running and comparing multiple experiments across datasets and optimizers, use the Jupyter notebook:
   ```
   Experiments.ipynb
   ```
   This notebook allows you to monitor training, compare learning curves and scatter plots, and view main regression metrics for all runs.

## Results and Analysis

- All experiment results, plots, and metrics are generated and saved automatically.
- See the notebook and the `final_results/` directory for output visualizations and summaries.
