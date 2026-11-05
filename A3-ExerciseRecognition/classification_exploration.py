"""Step 5: freeze validation choices, then evaluate retrained models on test."""

import hashlib
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from classification import DEFAULT_PARAMETERS, MODEL_GRIDS, make_classifier, tune_classifier
from features import BASELINE_FEATURES, SELECTED_EXTRA_FEATURES, add_my_features
from segmentation import adaptive_segment
from util import (build_feature_frames, load_manifest, select_recordings, split_recordings, viz_tree)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _freeze_selection(output, inputs, train_frames, fit, validation, features):
    fit_frames = select_recordings(train_frames, fit)
    val_frames = select_recordings(train_frames, validation)
    all_rows, all_predictions, models, balancing = [], [], {}, []
    for name in MODEL_GRIDS:
        print(f"Tuning {name} using fit/validation recordings only...", flush=True)
        rows, predictions, selected = tune_classifier(name, fit_frames, val_frames, features)
        default = next(row for row in rows if row["parameters"] == DEFAULT_PARAMETERS[name])
        models[name] = {
            "selected_parameters": selected["parameters"],
            "selected_candidate": selected["candidate"],
            "selected_validation_window_accuracy": selected["validation_window_accuracy"],
            "default_parameters": DEFAULT_PARAMETERS[name],
            "default_candidate": default["candidate"],
            "default_validation_window_accuracy": default["validation_window_accuracy"],
        }
        for variant, row in (("default", default), ("selected", selected)):
            values = predictions[predictions.candidate == row["candidate"]]
            report = classification_report(values.true, values.predicted, output_dict=True, zero_division=0)
            _write_json(output / f"report_{name}_{variant}_validation.json", report)
        # Matched comparison: change only weighting at the winning other settings.
        for weight in (None, "balanced"):
            parameters = {**selected["parameters"], "class_weight": weight}
            row = next(row for row in rows if row["parameters"] == parameters)
            values = predictions[predictions.candidate == row["candidate"]]
            report = classification_report(values.true, values.predicted, output_dict=True, zero_division=0)
            for exercise in sorted(validation.exercise.unique()):
                balancing.append({"model": name, "class_weight": str(weight), "exercise": exercise,
                                  **report[exercise], "validation_window_accuracy": row["validation_window_accuracy"]})
        all_rows.extend(rows)
        all_predictions.append(predictions)

    grid = pd.DataFrame(all_rows)
    grid["parameters"] = grid.parameters.map(json.dumps)
    grid.to_csv(output / "validation_grid.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(output / "validation_predictions_all_candidates.csv", index=False)
    pd.DataFrame(balancing).to_csv(output / "class_weight_validation_by_exercise.csv", index=False)
    selected_family = max(models, key=lambda name: models[name]["selected_validation_window_accuracy"])
    selection = {"selection_rule": "highest validation window accuracy; first declared candidate on exact ties",
                 "inputs": inputs, "grids": MODEL_GRIDS, "models": models,
                 "selected_model_family": selected_family,
                 "test_used_for_selection": False}
    # This artifact exists before any test signal is loaded. Later runs reuse it.
    _write_json(output / "frozen_selection.json", selection)
    print(f"Selection frozen before test evaluation. Final model family: {selected_family}", flush=True)
    return selection


def run_classification_exploration(output_dir="results/step5_classification"):
    output = Path(output_dir)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    feature_summary = json.loads(Path("results/step4_features/summary.json").read_text())
    features = feature_summary["selected_features"]
    assert features == BASELINE_FEATURES + SELECTED_EXTRA_FEATURES
    assert feature_summary["selected_segmentation"] == "adaptive"
    frames_path = Path(feature_summary["selected_frames_path"])
    train_frames = pd.read_csv(frames_path)
    train, test = split_recordings(load_manifest(), test_size=0.3, random_state=42)
    fit, validation = split_recordings(train, test_size=0.25, random_state=42)
    assert set(train_frames.recording) == set(train.path)
    assert set(train_frames.recording).isdisjoint(test.path)
    assert set(fit.path).isdisjoint(validation.path)
    assert np.isfinite(train_frames[features].to_numpy()).all()

    inputs = {
        "window_sec": feature_summary["selected_window_sec"], "features": features,
        "segmentation": "adaptive", "random_state": 42,
        "training_frames_sha256": hashlib.sha256(frames_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path("data/manifest.csv").read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ("util.py", "features.py", "segmentation.py", "classification.py",
                                       "classification_exploration.py")},
    }
    selection_path = output / "frozen_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["inputs"] != inputs or selection["grids"] != MODEL_GRIDS:
            raise ValueError("The frozen model-selection inputs changed. Review this explicitly; do not retune using test results.")
        print("Reusing the previously frozen validation choices.", flush=True)
    else:
        selection = _freeze_selection(output, inputs, train_frames, fit, validation, features)

    frozen_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        assert summary["frozen_selection_sha256"] == frozen_hash
        print("Reusing the completed test evaluation; no settings are changed.", flush=True)
        return summary

    print("Building test windows only after selection was frozen...", flush=True)
    test_frames = build_feature_frames(test, inputs["window_sec"], adaptive_segment, add_my_features)
    assert set(test_frames.recording) == set(test.path)
    assert set(test_frames.recording).isdisjoint(train.path)
    assert np.isfinite(test_frames[features].to_numpy()).all()
    test_frames.to_csv(output / "features_test.csv", index=False)
    predictions = test_frames[["recording", "exercise"]].copy()
    rows, model_paths, figures = [], {}, {}
    classes = sorted(train.exercise.unique())
    selected_models = {}
    for name, choices in selection["models"].items():
        for variant in ("default", "selected"):
            parameters = choices[f"{variant}_parameters"]
            model = make_classifier(name, parameters)
            model.fit(train_frames[features], train_frames.exercise)
            pred = model.predict(test_frames[features])
            column = f"{name}_{variant}"
            predictions[column] = pred
            report = classification_report(test_frames.exercise, pred, output_dict=True, zero_division=0)
            _write_json(output / f"report_{column}_test.json", report)
            rows.append({"model": name, "variant": variant, "parameters": parameters,
                         "validation_window_accuracy": choices[f"{variant}_validation_window_accuracy"],
                         "test_window_accuracy": float(accuracy_score(test_frames.exercise, pred)),
                         "test_macro_f1": report["macro avg"]["f1-score"],
                         "training_windows": len(train_frames), "test_windows": len(test_frames)})
            filename = output / "models" / f"{column}.joblib"
            joblib.dump(model, filename)
            model_paths[column] = str(filename)
            if variant == "selected":
                selected_models[name] = model
                fig, ax = plt.subplots(figsize=(10, 9))
                ConfusionMatrixDisplay.from_predictions(
                    test_frames.exercise, pred, labels=classes, ax=ax, xticks_rotation="vertical", colorbar=False
                )
                ax.set_title(f"{name.replace('_', ' ').title()}: test windows")
                fig.tight_layout()
                filename = output / "figures" / f"{name}_test_confusion.png"
                fig.savefig(filename, dpi=140)
                plt.close(fig)
                figures[f"{name}_confusion"] = str(filename)

    predictions.to_csv(output / "window_predictions_test.csv", index=False)
    table = pd.DataFrame(rows)
    table["parameters"] = table.parameters.map(json.dumps)
    table.to_csv(output / "model_comparison.csv", index=False)

    # Use the supplied helper unchanged, putting its dt.png in the results folder.
    original_dir = Path.cwd()
    figure_dir = (output / "figures").resolve()
    try:
        os.chdir(figure_dir)
        model = selected_models["decision_tree"]
        viz_tree(model, train_frames[features], list(model.classes_), max_depth=3)
        plt.close("all")
    finally:
        os.chdir(original_dir)
    figures["decision_tree_structure"] = str(output / "figures" / "dt.png")

    importance = pd.Series(selected_models["random_forest"].feature_importances_, index=features).sort_values()
    importance.rename("importance").to_csv(output / "forest_feature_importances.csv", index_label="feature")
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot.barh(ax=ax)
    ax.set_xlabel("Impurity-based importance")
    ax.set_title("Selected Random Forest, trained on all training windows")
    fig.tight_layout()
    filename = output / "figures" / "forest_feature_importances.png"
    fig.savefig(filename, dpi=140)
    plt.close(fig)
    figures["forest_feature_importances"] = str(filename)

    per_class = []
    for name in selection["models"]:
        report = json.loads((output / f"report_{name}_selected_test.json").read_text())
        for exercise in classes:
            per_class.append({"model": name, "exercise": exercise, **report[exercise]})
    pd.DataFrame(per_class).to_csv(output / "selected_models_test_by_exercise.csv", index=False)
    summary = {
        "scope": "Step 5: frozen validation selection, final test-window evaluation",
        "frozen_selection_sha256": frozen_hash,
        "selected_model_family": selection["selected_model_family"],
        "selection": selection["models"], "model_comparison": rows,
        "features": features, "window_sec": inputs["window_sec"], "segmentation": "adaptive",
        "fit_recordings": len(fit), "validation_recordings": len(validation),
        "training_recordings": len(train), "test_recordings": len(test),
        "model_paths": model_paths, "test_frames_path": str(output / "features_test.csv"),
        "figures": figures,
    }
    _write_json(summary_path, summary)
    print(table[["model", "variant", "validation_window_accuracy", "test_window_accuracy"]].to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    run_classification_exploration()
