#!/usr/bin/env python3
"""
Hyperparameter search utilities for neural network and protein transformer.
Splits the training data into train/validation (80/20), evaluates grid search
combinations defined in `hparam_config.json`, and saves the best configuration
for each model under `src/protein/hparam_results/`.
"""
import json
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

try:
    from dataset import ProteinDataLoader
    from model import NeuralNetworkClassifier, ProteinTransformerClassifier
except ImportError:
    from .dataset import ProteinDataLoader
    from .model import NeuralNetworkClassifier, ProteinTransformerClassifier

DATA_PATH = Path('src/data/protein/proteomic_encoder_train.csv')
CONFIG_PATH = Path('src/protein/hparam_config.json')
OUTPUT_DIR = Path('src/protein/hparam_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_SPLIT = 0.2
RANDOM_STATE = 42


def iterate_grid(grid):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        yield params


def to_numpy(data):
    if hasattr(data, "to_numpy"):
        return data.to_numpy()
    return np.asarray(data)


def evaluate_model(model_cls, params, X_train, y_train, X_val, y_val):
    X_train_np = to_numpy(X_train)
    y_train_np = to_numpy(y_train)
    X_val_np = to_numpy(X_val)
    y_val_np = to_numpy(y_val)

    tuned_params = dict(params)
    tuned_params['patience'] = 10

    model = model_cls(**tuned_params, random_state=RANDOM_STATE)
    model.fit(X_train_np, y_train_np)
    preds = model.predict(X_val_np)
    return balanced_accuracy_score(y_val_np, preds)


def run_search():
    loader = ProteinDataLoader(
        data_path=str(DATA_PATH),
        label_col='research_group',
        id_col='RID',
        random_state=RANDOM_STATE
    )
    X_train_raw, y_train, _, _, _ = loader.get_train_test_split(
        train_path=str(DATA_PATH),
        test_path=None
    )

    X_search, X_val, y_search, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=VAL_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    results_summary = {}

    searches = [
        ('neural_network', NeuralNetworkClassifier, cfg['neural_network']),
        ('protein_transformer', ProteinTransformerClassifier, cfg['protein_transformer']),
    ]

    for name, model_cls, grid in searches:
        best_score = -np.inf
        best_params = None
        records = []

        for params in iterate_grid(grid):
            score = evaluate_model(model_cls, params, X_search, y_search, X_val, y_val)
            records.append({'params': params, 'bal_acc': float(score)})
            if score > best_score:
                best_score = score
                best_params = params

        model_result = {
            'best_params': best_params,
            'best_score_bal_acc': float(best_score),
            'records': records,
        }
        results_summary[name] = model_result

        out_file = OUTPUT_DIR / f"{name}_search.json"
        with open(out_file, 'w') as f:
            json.dump(model_result, f, indent=2)

    summary_path = OUTPUT_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)


if __name__ == '__main__':
    run_search()