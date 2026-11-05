"""Classifier construction and fit/validation tuning for the fixed feature set."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier


MODEL_GRIDS = {
    "decision_tree": {
        "max_depth": [3, 5, 8, 12, None],
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced"],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 3],
        "class_weight": [None, "balanced"],
    },
}
DEFAULT_PARAMETERS = {
    "decision_tree": {"max_depth": 5, "criterion": "entropy", "class_weight": None},
    "random_forest": {"n_estimators": 100, "max_depth": None,
                      "min_samples_leaf": 1, "class_weight": None},
}


def make_classifier(name, parameters):
    """Use a fixed seed so the declared comparisons can be reproduced."""
    constructors = {"decision_tree": DecisionTreeClassifier, "random_forest": RandomForestClassifier}
    return constructors[name](random_state=42, **parameters)


def tune_classifier(name, fit_frames, val_frames, features):
    """Select by validation window accuracy; test frames are never an input."""
    assert set(fit_frames.recording).isdisjoint(val_frames.recording)
    rows, predictions = [], []
    for index, parameters in enumerate(ParameterGrid(MODEL_GRIDS[name])):
        model = make_classifier(name, parameters)
        model.fit(fit_frames[features], fit_frames.exercise)
        predicted = model.predict(val_frames[features])
        rows.append({"model": name, "candidate": index, "parameters": parameters,
                     "validation_window_accuracy": float(accuracy_score(val_frames.exercise, predicted)),
                     "validation_macro_f1": float(f1_score(val_frames.exercise, predicted, average="macro", zero_division=0))})
        predictions.append(pd.DataFrame({"model": name, "candidate": index,
                                        "window": np.arange(len(val_frames)),
                                        "recording": val_frames.recording, "true": val_frames.exercise,
                                        "predicted": predicted}))
    # Python max keeps the first item on an exact tie, as declared in the log.
    selected = max(rows, key=lambda row: row["validation_window_accuracy"])
    print(f"{name}: best validation accuracy {selected['validation_window_accuracy']:.2%}; "
          f"{selected['parameters']}", flush=True)
    return rows, pd.concat(predictions, ignore_index=True), selected
