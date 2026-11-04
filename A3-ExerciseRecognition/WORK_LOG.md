# Exercise Recognition: Work Log

This is our running record of decisions, experiments, and findings. We will use
it to write the exploration report. Results here are our measurements; numbers
in the rubric's example tables are only examples.

## Project goal

For each phone recording, find the exercise set, identify the exercise, and
count its repetitions. We need to explain what works, what fails, and why.

## Progress

- [x] Step 1: Set up the environment and measure the provided baselines.
- [ ] Step 2: Compare segmentation methods and inspect at least six recordings.
- [ ] Step 3: Improve and compare repetition counters.
- [ ] Step 4: Compare window lengths and features.
- [ ] Step 5: Tune and compare two classifier types.
- [ ] Step 6: Evaluate final predictions for whole recordings.
- [ ] Step 7: Investigate and illustrate the top two failure modes.
- [ ] Step 8: Complete the notebook, report, and submission ZIP.

## Rules for our experiments

- Keep the supplied `util.py` unchanged.
- Use the prescribed split with `random_state=42`: 301 training recordings and
  129 test recordings. Training is divided into 225 fit and 76 validation
  recordings. Fit is for learning; validation is for choosing settings.
- Keep every recording entirely in one split, even when it creates many windows.
- Use training/validation results to make decisions. Save test evaluation for
  the final comparisons after our settings are fixed.
- Change one design choice at a time so we can explain its effect.
- Record settings, numbers, and limitations. Small validation gains may be noise.
- Repetition evaluation uses the true exercise label unless we explicitly say
  that it uses the classifier's prediction.

## Initial repository review

We read both rubric versions, every notebook cell, and `util.py`. We checked the
whole manifest and inspected sample recordings covering all eight exercises and
all five placements.

- There are 430 recordings, and every manifest path exists.
- The CSV columns are `time`, `seconds_elapsed`, `z`, `y`, and `x`.
- The labels are exercise, reported repetition count, and phone placement.
- Sample rates vary: one inspected recording was about 58 Hz, while most
  inspected recordings were about 100 Hz. Use each recording's estimated rate.
- Exercise classes are uneven: 19 lat raises versus 106 squats, for example.
- Phone handling can resemble exercise. Some placements may barely register
  movement, and reported repetition counts may be slightly wrong.

## Step 1: Setup and baseline measurements (2026-11-04)

Status: complete. The notebook executed successfully and its outputs are saved.

We will first measure the supplied methods without tuning them. For counting,
we will use all 301 training recordings, as the notebook requests. For
classification, we will train on fit recordings and evaluate on validation
recordings. We will not evaluate predictions on the test set at this stage.

The initial classification window is 4 seconds, with the supplied 50% overlap.
This is a starting point from the notebook's examples, not a chosen best length.
We will compare lengths properly in Step 4.

### Environment and how to rerun

The project uses an isolated `.venv` in the repository root. It was created with
Python 3.14.3. `requirements.txt` records the versions used for this baseline.
The environment is excluded from Git; source files and measured results are kept.

From the repository root, recreate the environment if needed:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r A3-ExerciseRecognition/requirements.txt
```

To run the baseline and regenerate its saved CSV/JSON results:

```bash
.venv/bin/python A3-ExerciseRecognition/baseline.py
```

To execute the notebook and save its displayed outputs:

```bash
.venv/bin/python A3-ExerciseRecognition/execute_notebook.py
```

To work interactively, launch JupyterLab with the project environment and open
the assignment notebook:

```bash
.venv/bin/jupyter lab --notebook-dir=A3-ExerciseRecognition
```

The notebook is still a starter for Steps 2–8. Running it now measures the
baselines; the later TODOs and final-test tables remain unfinished.

### Baseline settings

| Part | What we used |
|---|---|
| Preprocessing | Supplied acceleration magnitude and 5 Hz low-pass filter |
| Segmentation | First to last sample above a fixed motion threshold (1-second moving standard deviation > 0.5 m/s²); supplied fallback if nothing crosses it |
| Counting | Peaks above the mean of the filtered magnitude, at least 1 second apart |
| Features | Mean, maximum, median, minimum, 25th percentile, 75th percentile, and standard deviation of filtered magnitude |
| Classification windows | 4 seconds, with 50% overlap |
| Decision Tree | `max_depth=5`, `criterion='entropy'`, `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42`; other settings are library defaults |

Neither model has been tuned. The models learn from fit windows only. Validation
accuracy counts correct windows, so a long recording has more influence than a
short one. Whole-recording voting comes later in Step 6.

### How to read the counting metrics

- **MAE:** average absolute difference between counted and reported repetitions.
  Smaller is better. An MAE of 4 means an average error of four repetitions.
- **Within ±2:** fraction of recordings with at most two repetitions of error.
  Bigger is better. In saved files, `0.25` means 25%.
- **Bias:** average of counted minus reported repetitions. Positive means
  overcounting overall; negative means undercounting. Opposite errors can cancel,
  so bias near zero does not necessarily mean accurate counting.

### Files saved for later comparisons

`results/step1_baseline/` contains:

- `summary.json`: settings, package versions, overall metrics, and the checksum
  of the unchanged `util.py`.
- Split manifests and `split_counts.csv`: the exact recordings in each split.
  Saving the test membership does not evaluate test signals or predictions.
- `rep_predictions_train.csv`: counted and reported repetitions for every
  training recording.
- `rep_*_by_exercise.csv` and `rep_*_by_placement.csv`: counting breakdowns for
  training, fit, and validation recordings.
- `features_train_4s.csv`: the baseline windows and seven features, for training
  recordings only.
- `window_predictions_validation.csv` and `classification_*_validation.json`:
  predictions and per-exercise classification statistics for the two models.

These files are regenerated when the baseline runs. Keep their directory as the
Step 1 reference; save later experiments in separate directories.

### Results: repetition counting

All rows below use the same supplied segmentation and supplied counter. The
training row includes both fit and validation, so these rows are not independent
experiments. We include the smaller splits to support later comparisons.

| Recordings evaluated | Number | MAE (reps) | Within ±2 reps | Bias (reps) |
|---|---:|---:|---:|---:|
| All training | 301 | 7.98 | 19.60% | +4.61 |
| Fit only | 225 | 7.98 | 21.33% | +4.80 |
| Validation only | 76 | 7.99 | 14.47% | +4.04 |

Breakdown by exercise on **all training recordings**:

| Exercise | Recordings | MAE (reps) | Within ±2 reps | Bias (reps) |
|---|---:|---:|---:|---:|
| Ab crunches | 20 | 5.55 | 35.00% | +4.05 |
| Bicep curls | 35 | 7.89 | 17.14% | +2.80 |
| Jumping jacks | 61 | 6.13 | 22.95% | -5.61 |
| Lat raises | 13 | 8.62 | 15.38% | +6.77 |
| Lunges | 29 | 10.72 | 10.34% | +10.72 |
| Push-ups | 41 | 6.22 | 12.20% | +5.10 |
| Sit-ups | 28 | 10.14 | 17.86% | +10.14 |
| Squats | 74 | 9.18 | 22.97% | +8.91 |

Breakdown by placement on **all training recordings**:

| Placement | Recordings | MAE (reps) | Within ±2 reps | Bias (reps) |
|---|---:|---:|---:|---:|
| Hand | 141 | 7.02 | 29.08% | +2.95 |
| Trouser | 128 | 8.83 | 9.38% | +6.08 |
| Jacket | 17 | 8.24 | 17.65% | +7.53 |
| Armband | 9 | 12.33 | 0.00% | +4.11 |
| Bag/backpack | 6 | 5.17 | 50.00% | +4.83 |

**What this tells us:** the baseline is often inaccurate and overcounts overall.
Lunges and sit-ups have large average errors. Jumping jacks are undercounted on
average, so we should not assume every exercise needs fewer detected peaks.
The causes still need to be checked against signal plots in later steps.

Hand recordings have lower error than trouser recordings in this training set.
That does not prove placement alone caused the difference: the mix of exercises
also differs by placement. Armband and bag/backpack have very few examples, so
their percentages are especially uncertain.

### Results: classification

The feature builder produced **3,677 fit windows** from 225 recordings and
**1,156 validation windows** from 76 recordings. Both models used exactly the
same seven features and windows.

| Model | Validation window accuracy |
|---|---:|
| Default Decision Tree | 41.87% |
| Default Random Forest | 46.80% |

**What this tells us:** the Random Forest was 4.93 percentage points higher on
this validation split. This is an initial comparison, not a final model choice.
The validation set is small, and the overlapping windows are not independent
recordings. We still need feature exploration, tuning, and whole-recording
evaluation before drawing stronger conclusions.

### Checks completed

- All 430 manifest paths exist and are unique.
- The prescribed split sizes are correct and all eight exercises appear in
  every split. Fit and validation together exactly cover the training set.
- Fit, validation, and test share no recording paths.
- Every fit and validation recording produced at least one feature window.
- All seven feature columns contain finite numbers; no NaN or infinity values.
- There is one counting result per training recording, with nonnegative counts.
- The notebook ran from top to bottom without execution errors. Later TODOs
  remain placeholders; this does not mean the assignment is finished.
- The installed packages have no broken dependency requirements.
- `util.py` is unchanged, and baseline predictions do not include test recordings.

The split is stratified by **exercise**, not placement. All six bag/backpack
recordings happen to be in training; there are none in the test split. Later
tables should describe the placements actually present rather than inventing a
test result for this placement.

### Material for the report and next step

These baseline settings and development results will help us explain the
starting point in report Sections 1–3. They are **not** the final test metrics
that those sections require. Save final test evaluation until the methods and
settings have been fixed.

Next is **Step 2: segmentation**. Inspect at least six fit recordings across
exercises and placements, compare the baseline with our own method, and keep
examples where each fails. Use the same counter and classifier settings when
measuring the effect of segmentation, so we can identify what caused a change.
