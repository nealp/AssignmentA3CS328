"""Step 6: evaluate fixed majority votes for the frozen Step 5 classifiers."""

import hashlib
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix

from recording_predictions import predict_recordings
from util import load_manifest, split_recordings


def run_recording_exploration(output_dir="results/step6_recordings"):
    output = Path(output_dir)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    step5 = Path("results/step5_classification")
    summary5 = json.loads((step5 / "summary.json").read_text())
    frozen_path = step5 / "frozen_selection.json"
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == summary5["frozen_selection_sha256"]
    frozen = json.loads(frozen_path.read_text())
    for name, digest in frozen["inputs"]["source_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    train, test = split_recordings(load_manifest(), test_size=0.3, random_state=42)
    frames = pd.read_csv(summary5["test_frames_path"])
    saved_windows = pd.read_csv(step5 / "window_predictions_test.csv")
    assert set(frames.recording) == set(test.path)
    assert set(frames.recording).isdisjoint(train.path)
    assert frames[["recording", "exercise"]].equals(saved_windows[["recording", "exercise"]])
    features = summary5["features"]
    classes = sorted(test.exercise.unique())
    model_rows, per_class, figures, model_hashes = [], [], {}, {}
    final_family = summary5["selected_model_family"]
    for name in ("decision_tree", "random_forest"):
        model_path = Path(summary5["model_paths"][f"{name}_selected"])
        model_hashes[name] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        model = joblib.load(model_path)
        assert list(model.feature_names_in_) == features
        window_pred = model.predict(frames[features])
        assert np.array_equal(window_pred, saved_windows[f"{name}_selected"].to_numpy())
        recordings = predict_recordings(model, frames, features)
        assert len(recordings) == len(test) == 129
        assert recordings.recording.is_unique and set(recordings.recording) == set(test.path)
        truth = test.set_index("path").exercise
        assert recordings.set_index("recording").true.equals(truth.reindex(recordings.recording))
        recordings.to_csv(output / f"{name}_recordings_test.csv", index=False)

        windows = frames[["recording", "exercise"]].assign(predicted=window_pred)
        diagnostics = []
        for recording, group in windows.groupby("recording", sort=False):
            counts = group.predicted.value_counts()
            winning_count = int(counts.max())
            tied = sorted(counts[counts == winning_count].index.tolist())
            true = group.exercise.iloc[0]
            diagnostics.append({
                "recording": recording, "true": true, "predicted": tied[0],
                "windows": len(group), "winning_votes": winning_count,
                "winning_share": winning_count / len(group), "tie": len(tied) > 1,
                "tied_labels": ", ".join(tied),
                "correct_window_fraction": float(group.predicted.eq(true).mean()),
            })
        diagnostics = pd.DataFrame(diagnostics)
        assert recordings.predicted.tolist() == diagnostics.predicted.tolist()
        diagnostics.to_csv(output / f"{name}_vote_diagnostics_test.csv", index=False)
        report = classification_report(recordings.true, recordings.predicted, labels=classes,
                                       output_dict=True, zero_division=0)
        (output / f"{name}_recording_report_test.json").write_text(json.dumps(report, indent=2) + "\n")
        for exercise in classes:
            per_class.append({"model": name, "exercise": exercise, **report[exercise]})
        matrix = confusion_matrix(recordings.true, recordings.predicted, labels=classes)
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(output / f"{name}_recording_confusion.csv")
        model_rows.append({
            "model": name, "test_recordings": len(recordings), "test_windows": len(frames),
            "window_accuracy": float(accuracy_score(frames.exercise, window_pred)),
            "mean_recording_window_accuracy": float(diagnostics.correct_window_fraction.mean()),
            "recording_accuracy": float(accuracy_score(recordings.true, recordings.predicted)),
            "correct_recordings": int(recordings.true.eq(recordings.predicted).sum()),
            "recording_macro_f1": report["macro avg"]["f1-score"],
            "tied_recordings": int(diagnostics.tie.sum()),
        })
        fig, ax = plt.subplots(figsize=(10, 9))
        ConfusionMatrixDisplay(matrix, display_labels=classes).plot(
            ax=ax, xticks_rotation="vertical", colorbar=False
        )
        title = f"{name.replace('_', ' ').title()}: test recordings"
        if name == final_family:
            title = "Final pipeline: " + title
        ax.set_title(title)
        fig.tight_layout()
        figure_path = output / "figures" / f"{name}_recording_confusion.png"
        fig.savefig(figure_path, dpi=140)
        plt.close(fig)
        figures[name] = str(figure_path)
        if name == final_family:
            recordings.to_csv(output / "final_recording_predictions.csv", index=False)
            errors = recordings[recordings.true != recordings.predicted].merge(
                test[["path", "placement", "reps"]], left_on="recording", right_on="path"
            ).drop(columns="path")
            errors.to_csv(output / "final_classification_errors.csv", index=False)

    pd.DataFrame(model_rows).to_csv(output / "recording_comparison.csv", index=False)
    pd.DataFrame(per_class).to_csv(output / "recording_recall_by_exercise.csv", index=False)
    summary = {
        "scope": "Step 6: fixed majority voting on Step 5 test-window predictions; no tuning",
        "voting_rule": "most frequent window label; exact ties resolved alphabetically",
        "selected_model_family": final_family, "comparison": model_rows,
        "figures": figures, "final_confusion_figure": figures[final_family],
        "frozen_selection_sha256": summary5["frozen_selection_sha256"],
        "model_sha256": model_hashes,
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ("recording_predictions.py", "recording_exploration.py")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(model_rows).to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    run_recording_exploration()
