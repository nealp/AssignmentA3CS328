# Exercise Recognition: Work Log

This is our plain-language record of what we tried, what happened, and what we
need to explain in the report. Numbers below are our measurements, not the
example numbers in the rubric.

## Progress

- [x] Step 1: Set up and measure the supplied baselines.
- [x] Step 2: Compare segmentation methods and inspect recordings.
- [x] Step 3: Improve and compare repetition counters (development comparisons).
- [x] Step 4: Compare window lengths and features.
- [x] Step 5: Tune and compare two classifier types.
- [x] Step 6: Evaluate predictions for whole recordings.
- [ ] Step 7: Investigate the top two failure modes.
- [ ] Step 8: Finish the notebook, report, and submission ZIP.

## Current project structure

The user prefers separate Python modules. We restored that structure before
starting Step 3. The notebook remains the assignment narrative and displays
our tables and figures; the modules contain reusable methods and experiments.

- `util.py`: supplied helpers, unchanged.
- `baseline.py`: reproduce Step 1.
- `segmentation.py`: segmentation methods.
- `segmentation_exploration.py`: Step 2 comparisons and plots.
- `rep_counting.py` and `rep_counting_exploration.py`: Step 3 work.
- `features.py` and `feature_exploration.py`: Step 4 features and comparisons.
- `classification.py` and `classification_exploration.py`: Step 5 model tuning
  and frozen test evaluation.
- `recording_predictions.py` and `recording_exploration.py`: Step 6 voting and
  recording-level evaluation.
- `results/step1_baseline/`, `results/step2_segmentation/`, and
  `results/step3_rep_counting/`: settings, predictions, tables, and figures.

Step 4 outputs are under `results/step4_features/` with the same organization.
Step 5 outputs are under `results/step5_classification/`, including saved models.
Step 6 outputs are under `results/step6_recordings/`.

The extra modules and results folder are organizational choices, not required
filenames in the rubric. Include our needed Python files with the notebook in
the final submission. The optional `execute_notebook.py` stays deleted.

## Experiment rules

- Use the prescribed split with `random_state=42`: 301 training and 129 test
  recordings. Training divides into 225 fit and 76 validation recordings.
- Keep all windows of a recording in the same split.
- Learn on fit recordings and choose settings using training/validation results.
  Reserve test evaluation for the final comparisons.
- Change one design choice at a time to understand its effect.
- Use each recording's estimated sample rate.
- Repetition evaluation uses the true exercise label unless explicitly stated
  otherwise. This differs from using a possibly incorrect predicted exercise.
- Treat small validation differences cautiously; 76 recordings is a small set.

## Initial review

We read both rubric versions, every notebook cell, and `util.py`. We checked the
manifest and inspected sample CSVs from all eight exercises and five placements.
All 430 listed recordings exist. Their raw columns are `time`,
`seconds_elapsed`, `z`, `y`, and `x`.

Classes are uneven: lat raises has 19 recordings and squats has 106. Most
inspected sample rates were about 100 Hz, but one was about 58 Hz. Phone handling
can resemble exercise, some placements barely move during an exercise, and
self-reported repetition counts can be slightly wrong.

## Environment

We created an isolated `.venv` with Python 3.14.3. `requirements.txt` records the
package versions. These are setup aids, not report deliverables.

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r A3-ExerciseRecognition/requirements.txt
.venv/bin/jupyter lab --notebook-dir=A3-ExerciseRecognition
```

Open the notebook and use **Restart Kernel and Run All Cells**, then save it.
The later assignment sections still have TODOs; successful execution of the
current notebook does not mean the assignment is finished.

## Step 1: Supplied baselines

### What we ran

We used the supplied methods without tuning:

| Part | Settings |
|---|---|
| Preprocessing | Supplied magnitude calculation and 5 Hz low-pass filter |
| Segmentation | First to last crossing of a fixed threshold: 1-second moving standard deviation > 0.5 m/s² |
| Counting | Peaks above mean filtered magnitude, at least 1 second apart |
| Features | Mean, maximum, median, minimum, 25th/75th percentiles, standard deviation |
| Windows | 4 seconds, 50% overlap; a starting point, not a tuned choice |
| Decision Tree | `max_depth=5`, `criterion='entropy'`, `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42`, other supplied defaults |

### Counting results

MAE is average absolute counting error; smaller is better. Within ±2 is the
percentage of recordings within two reported reps; larger is better. Bias is
counted minus reported reps: positive means overcounting overall. Opposite
errors can cancel, so small bias alone does not mean accurate counting.

| Evaluated split | Recordings | MAE (reps) | Within ±2 | Bias (reps) |
|---|---:|---:|---:|---:|
| All training | 301 | 7.98 | 19.60% | +4.61 |
| Fit | 225 | 7.98 | 21.33% | +4.80 |
| Validation | 76 | 7.99 | 14.47% | +4.04 |

Training includes fit and validation; these rows are not independent experiments.

Baseline counting by exercise, on all training recordings:

| Exercise | Recordings | MAE | Within ±2 | Bias |
|---|---:|---:|---:|---:|
| Ab crunches | 20 | 5.55 | 35.00% | +4.05 |
| Bicep curls | 35 | 7.89 | 17.14% | +2.80 |
| Jumping jacks | 61 | 6.13 | 22.95% | -5.61 |
| Lat raises | 13 | 8.62 | 15.38% | +6.77 |
| Lunges | 29 | 10.72 | 10.34% | +10.72 |
| Push-ups | 41 | 6.22 | 12.20% | +5.10 |
| Sit-ups | 28 | 10.14 | 17.86% | +10.14 |
| Squats | 74 | 9.18 | 22.97% | +8.91 |

Baseline counting by placement, on all training recordings:

| Placement | Recordings | MAE | Within ±2 | Bias |
|---|---:|---:|---:|---:|
| Hand | 141 | 7.02 | 29.08% | +2.95 |
| Trouser | 128 | 8.83 | 9.38% | +6.08 |
| Jacket | 17 | 8.24 | 17.65% | +7.53 |
| Armband | 9 | 12.33 | 0.00% | +4.11 |
| Bag/backpack | 6 | 5.17 | 50.00% | +4.83 |

The baseline overcounts overall, especially lunges and sit-ups. Jumping jacks
are undercounted on average, so a universal change to reduce peaks may hurt.
Hand recordings have lower error than trouser recordings, but the mix of
exercises also differs. Small placement groups give uncertain percentages.

### Classification results

Both models learned from 3,677 fit windows and were evaluated on 1,156
validation windows, using exactly the same features and windows.

| Default model | Validation window accuracy |
|---|---:|
| Decision Tree | 41.87% |
| Random Forest | 46.80% |

The Random Forest was 4.93 percentage points higher on this split. Neither model
was tuned. Window accuracy gives longer recordings more influence because they
produce more windows; recording-level voting comes later.

## Step 2: Segmentation

### Plan and implementation

We compared three approaches while keeping the counter, features, window
length, and classifier settings unchanged:

1. Supplied fixed-threshold segmentation.
2. Protocol trimming: remove five seconds at each end, unless the recording is
   too short. This assumes the person followed the requested rest protocol.
3. Our adaptive method, implemented directly in `find_exercise_segment`.

Adaptive settings were fixed before aggregate scoring; we did not search a grid:

- Compute moving standard deviation over one second.
- Quiet/active levels are its 10th/90th percentiles.
- Threshold = quiet + 20% of their difference, with a minimum of 0.02 m/s².
- Join internal gaps of at most two seconds; discard bouts shorter than four.
- Keep the longest remaining bout and add 0.75 seconds at each boundary.
- If no sustained bout is found, retain the whole recording.

The method uses only signal and sample rate, not exercise, placement, reported
reps, filename, or split. There are no ground-truth boundary labels, so we use
plots and downstream performance together. Lower counting error alone can be
misleading if trimming removes real reps from an overcounting counter.

### Visual inspection

We selected one fit recording per exercise by sorted filename, then added
examples for missing placements. This gave ten recordings spanning all eight
exercises and all five placements, without selecting by performance. The
notebook displays their paths and baseline/adaptive overlays.

Observations:

- Jumping jacks `003_2026-09-13_23-59-47.csv` (Hand): adaptive ends near 37.13
  seconds; baseline continues to 42.40, including later isolated movement.
- Squats `002_2026-09-13_19-25-27.csv` (Trouser): adaptive excludes the initial
  burst, but still keeps a large final burst close to the exercise.
- Weak motion and missing rest periods remain difficult. The longest detected
  bout is not guaranteed to be the exercise set.

### Counting comparison: unchanged baseline counter

| Segmentation | Validation MAE | Validation within ±2 | Validation bias | Fit MAE | Fit within ±2 |
|---|---:|---:|---:|---:|---:|
| Baseline | 7.99 | 14.47% | +4.04 | 7.98 | 21.33% |
| Trim five seconds | 6.01 | 21.05% | +1.67 | 6.56 | 32.89% |
| Adaptive | 5.95 | 21.05% | +1.39 | 6.36 | 28.00% |

Adaptive reduced validation MAE by 2.04 reps versus baseline. Its difference
from fixed trimming is only 0.07 reps, and both counted 16 of 76 validation
recordings within two reps. That is not a meaningful overall winner.
All-training MAE was 7.98, 6.42, and 6.26 respectively.

### Classification comparison: unchanged features and models

| Segmentation | Fit windows | Validation windows | Decision Tree accuracy | Random Forest accuracy |
|---|---:|---:|---:|---:|
| Baseline | 3,677 | 1,156 | 41.87% | 46.80% |
| Trim five seconds | 2,889 | 882 | 44.90% | 49.55% |
| Adaptive | 2,670 | 887 | 49.94% | 48.93% |

Adaptive improved Decision Tree accuracy by 8.08 percentage points and Random
Forest accuracy by 2.13 points versus baseline. Fixed trim was slightly higher
for the Random Forest. Changing segmentation changes both window content and
window counts: these are comparisons on the same recordings, not identical
windows. Recording-level evaluation is still pending.

### Specific failure examples for the report

We inspected fit errors only to find failures. Both examples are plotted in the
notebook's segmentation visual-check section.

**Baseline failure:** `data/squats/002_2026-09-13_19-25-27.csv`, Trouser, 10
reported reps. Baseline keeps 0.63–36.30 seconds, including the initial burst
before the repeated lower-amplitude motion. Adaptive keeps 6.85–37.11 seconds.
With the same counter, counts change from 19 to 16, still above 10. The final
burst remains attached to the set. The isolated bursts look like phone handling,
but there is no video or labeled boundary to confirm the cause.

**Our failure:** `data/ab_crunches/002_2026-10-13_22-50-15.csv`, Hand, 27 reported
reps. Repeated motion continues across much of the roughly 42-second trace, but
its amplitude decreases. The adaptive threshold splits it into two bouts and
keeps only 0–20.66 seconds. Counts change from 29 with baseline segmentation to
14 with adaptive: absolute error worsens from 2 to 13. This illustrates a
segmentation error cutting activity, with no change to the counting algorithm.

Another fit recording, ab crunches `023_2026-10-19_03-49-58.csv` (Trouser), has
much stronger bursts at the ends than in the middle. Adaptive picks 0–7.02
seconds and counts 4 versus 14 reported. Weak exercise motion appears to be
obscured by handling.

The initial diagnostic run found multiple sustained bouts in 57 of 301 training
recordings and whole-recording fallback in 18. Multiple bouts are not necessarily
failures, but are useful cases to inspect.

### Working decision

Use adaptive segmentation for the next development steps. It adapts to the
signal, improves counting over baseline on fit and validation, and improves
both fixed classifiers on this split. It is not universally better than fixed
trimming. Revisit the comparison with the final counter and features before
choosing the final pipeline. No test-based choice has been made.

Future ideas, not yet implemented: account for gradual changes in activity
amplitude, or merge plausible exercise bouts without merging isolated handling.

## Verification and structure changes after Step 2

We verified split separation, finite feature values, one counting result per
training recording, valid segment boundaries, and no test predictions. Repeated
baseline results matched Step 1. Synthetic checks at 50 and 100 Hz covered short
pauses, isolated handling bursts, and flat-data fallback.

We briefly consolidated code into the notebook at the user's request and
verified identical results. The user then preferred the previous modular
structure, so we restored the modules and saved results from the backup. This
changes code organization, not the segmentation settings or prior measurements.
`util.py` and the data remain unchanged. The execution helper stays deleted.

## Step 3: Rep counting

Status: complete for development. The final test tables remain pending until
the full pipeline is fixed.

### 3.1 Plan before experiments

Keep adaptive segmentation fixed while comparing counters. Compare the supplied
counter with more strongly smoothed magnitude, the most active signed axis, and
the principal movement direction (PCA). A signed signal can avoid the two peaks
per rep created by acceleration magnitude. Stronger smoothing and peak prominence
can suppress small extra bumps.

Start with a small, declared set of settings on the 225 fit recordings. Select
using fit error, then check validation recordings and inspect example traces.
The counter must not use placement, reported reps, or filenames at prediction
time. Keep test evaluation for final reporting. Once a counter is chosen,
repeat segmentation comparisons with that same counter and document failures.

### 3.2 Initial candidate settings

Implemented signal extraction and peak detection in `rep_counting.py`.
We will compare six candidates: magnitude, strongest filtered axis, and PCA,
each with a 1.0 Hz or 1.5 Hz low-pass cutoff. Every candidate initially uses a
0.6-second minimum peak distance and prominence of 0.5 times its signal standard
deviation. These choices are recorded before scoring. Segmentation stays adaptive.

The filter is applied before selecting the strongest axis/PCA direction. PCA's
sign is fixed for repeatability. Signals with standard deviation below 0.01 m/s²
return zero rather than count tiny noise fluctuations. This can also miss very
weak real motion; it does not recover information an unsuitable placement lost.
The counter uses no exercise-specific settings at this stage.

### 3.3 First fit-only comparison and next tuning step

With adaptive segmentation fixed, the initial fit results were:

| Signal / cutoff | Fit MAE | Within ±2 | Bias |
|---|---:|---:|---:|
| Supplied magnitude counter (5 Hz) | 6.36 | 28.00% | +1.25 |
| Magnitude / 1.0 Hz | 6.61 | 30.22% | -2.40 |
| Magnitude / 1.5 Hz | 7.48 | 28.00% | +2.52 |
| Strongest axis / 1.0 Hz | 5.73 | 40.44% | +1.13 |
| Strongest axis / 1.5 Hz | 7.63 | 31.11% | +4.19 |
| PCA / 1.0 Hz | 5.33 | 43.11% | +0.71 |
| PCA / 1.5 Hz | 7.59 | 33.33% | +4.13 |

PCA at 1 Hz has the lowest fit MAE, so keep that signal and cutoff for the next
small fit-only comparison. Try peak distances of 0.6, 0.9, and 1.2 seconds, with
prominence of 0.5 or 1.0 times signal standard deviation. The 0.6/0.5 combination
is already measured. Longer distances may reject duplicate peaks but also miss
fast reps; larger prominence may reject bumps but miss weaker reps. Select by
fit MAE, then evaluate the frozen settings on validation. No validation counting
results have been examined for these new candidates yet.

### 3.4 Fit-selected settings frozen before validation

For PCA / 1 Hz, the fit-only distance/prominence comparison gave:

| Minimum distance (s) | Prominence × std | Fit MAE | Within ±2 | Bias |
|---|---:|---:|---:|---:|
| 0.6 | 0.5 | 5.33 | 43.11% | +0.71 |
| 0.6 | 1.0 | 4.59 | 50.67% | -1.87 |
| 0.9 | 0.5 | 5.31 | 43.11% | -0.50 |
| 0.9 | 1.0 | 4.81 | 50.22% | -2.57 |
| 1.2 | 0.5 | 5.76 | 37.33% | -3.24 |
| 1.2 | 1.0 | 5.76 | 40.89% | -4.48 |

The lowest fit MAE is **4.59**, using PCA, 1 Hz cutoff, 0.6-second minimum peak
distance, and prominence of **1.0 × signal standard deviation**. Freeze these
as `REP_SETTINGS` and use them in `count_reps`. No settings depend on placement
or exercise. The negative bias shows that some reps are now missed; higher
prominence is a tradeoff rather than a universal improvement.

`rep_counting_exploration.py` reproduces the two fit-selection stages, checks
that its choice matches the counter defaults, then evaluates validation. It
also repeats the segmentation comparisons with the chosen counter and produces
fit-recording figures with the exact detected peaks. Results will be saved in
`results/step3_rep_counting/`.

### 3.5 Validation results and segmentation recheck

The selected counter was frozen before this validation evaluation. All rows use
adaptive segmentation; `prom` is peak prominence as a multiple of signal std.

| Counter | Fit MAE | Validation MAE | Validation within ±2 | Validation bias |
|---|---:|---:|---:|---:|
| Supplied baseline | 6.36 | 5.95 | 21.05% | +1.39 |
| Magnitude, 1 Hz, prom 0.5 | 6.61 | 6.11 | 31.58% | -1.63 |
| Magnitude, 1.5 Hz, prom 0.5 | 7.48 | 6.99 | 17.11% | +3.33 |
| Strongest axis, 1 Hz, prom 0.5 | 5.73 | 3.88 | 47.37% | +0.86 |
| Strongest axis, 1.5 Hz, prom 0.5 | 7.63 | 6.13 | 36.84% | +4.00 |
| PCA, 1 Hz, prom 0.5 | 5.33 | 4.01 | 43.42% | +0.86 |
| PCA, 1.5 Hz, prom 0.5 | 7.59 | 6.34 | 34.21% | +4.29 |
| **Selected PCA, 1 Hz, prom 1.0** | **4.59** | **3.33** | **56.58%** | **-1.51** |

The table above summarizes the initial signal comparison plus the chosen
peak settings. All five additional distance/prominence candidates and their
validation metrics are preserved in `counter_comparison.csv` and shown in the
notebook. We did not make another settings change after reviewing validation.

The selected counter reduces validation MAE by **2.62 reps** versus the supplied
counter on the same segments. It counts **43 of 76** recordings within two reps,
versus **16 of 76**. Its remaining negative bias indicates undercounting.

To separate improvements from segmentation and counting, we also measured:

| Segmentation | Counter | Validation MAE | Within ±2 | Bias |
|---|---|---:|---:|---:|
| Whole recording | Baseline | 7.93 | 14.47% | +4.62 |
| Baseline | Baseline | 7.99 | 14.47% | +4.04 |
| Adaptive | Baseline | 5.95 | 21.05% | +1.39 |
| Baseline | Ours | 5.14 | 44.74% | -0.12 |
| Adaptive | Ours | 3.33 | 56.58% | -1.51 |
| Trim five seconds | Ours | 3.66 | 48.68% | -1.13 |

Both stages help on this split. With our counter fixed, adaptive segmentation
is 1.82 reps lower in MAE than baseline segmentation. Its MAE is only 0.33 lower
than protocol trimming, so keep the small-validation-set caveat. We retain
adaptive as the working segmentation; no segmentation settings were changed.

### 3.6 Hard exercises and placements

Selected counter plus adaptive segmentation, **validation recordings**:

| Exercise | Recordings | MAE | Within ±2 | Bias |
|---|---:|---:|---:|---:|
| Ab crunches | 5 | 4.20 | 40.00% | -1.80 |
| Bicep curls | 9 | 3.89 | 66.67% | -3.44 |
| Jumping jacks | 16 | 3.88 | 62.50% | -3.75 |
| Lat raises | 3 | 4.67 | 33.33% | -4.67 |
| Lunges | 7 | 1.86 | 71.43% | +1.29 |
| Push-ups | 10 | 3.50 | 40.00% | -0.70 |
| Sit-ups | 7 | 4.14 | 28.57% | -3.86 |
| Squats | 19 | 2.32 | 68.42% | +1.26 |

Lunges and squats improve substantially; upper-body exercises with weak phone
motion remain challenging. The three validation lat-raise recordings have high
error, but that group is too small to support a strong ranking.

| Placement | Validation recordings | MAE | Within ±2 | Bias |
|---|---:|---:|---:|---:|
| Hand | 39 | 3.13 | 64.10% | -1.18 |
| Trouser | 33 | 3.33 | 51.52% | -1.70 |
| Armband | 1 | 8.00 | 0.00% | -8.00 |
| Jacket | 1 | 4.00 | 0.00% | +4.00 |
| Bag/backpack | 2 | 4.50 | 50.00% | -4.50 |

There is too little validation data for the last three placements to generalize.
Across all training recordings, Hand MAE is 3.63, Trouser 4.75, and Armband 8.78
(nine recordings). Exercise mix also differs by placement. All-training MAE is
**4.27**, within-two is **52.16%**, and bias is **-1.78**. These aggregate figures
include the fit data used for selection; they are not new held-out evidence.

### 3.7 Signal plots and remaining failures

All three illustrations were selected from fit recordings after settings were
fixed, and show the selected segment, actual counted signal, and detected peaks.

- **Improvement:** lunges `019_2026-10-18_20-44-36.csv`, Trouser, 15 reported reps.
  With adaptive segmentation unchanged, the supplied magnitude counter gives
  **32**, while PCA gives **15**. The magnitude has extra bumps within each
  movement; the smoothed signed direction better follows full motion cycles.
- **Largest fit error:** jumping jacks `005_2026-10-04_17-25-59.csv`, Trouser,
  35 reported reps. Our counter gives **4**, versus **14** for the baseline on
  the same adaptive segment. The segment already excludes later activity.
  Within it, an initial large movement dominates the PCA signal/prominence
  scale, and the 1 Hz filter suppresses much of the faster repeated motion.
  This combines segmentation and rep-counting failures; improvement elsewhere
  does not make one cutoff suitable for all recordings.
- **Segmentation limit:** ab crunches `002_2026-10-13_22-50-15.csv`, Hand, 27
  reported reps. Adaptive keeps only about half the trace. Our counter gives
  **13**, versus **14** with the baseline counter on that segment. Changing the
  counter cannot recover repetitions outside the selected interval. With
  baseline segmentation, the same new counter gives **27**, matching the
  reported count; this supports attributing this particular error to segmentation.

The physical explanations are inferences from the traces, not verified video
annotations. Possible improvements are more robust prominence scaling, adapting
the cutoff to motion rate, and retaining valid exercise bouts. None has been
implemented after examining validation in this step.

### 3.8 Integration and verification

The notebook's Function 2 now calls `count_reps` in `rep_counting.py`. Its
rep-counting experiment cell calls `run_rep_counting_exploration` and displays
the comparisons, per-exercise/placement errors, and three example figures.
The required final-test tables remain unfilled until the final pipeline is fixed.

Reproduce the modular experiments from the repository root:

```bash
.venv/bin/python A3-ExerciseRecognition/baseline.py
.venv/bin/python A3-ExerciseRecognition/segmentation_exploration.py
.venv/bin/python A3-ExerciseRecognition/rep_counting_exploration.py
```

Counting results live in `results/step3_rep_counting/`: the candidate and
segmentation tables, predictions for every training recording, group breakdowns,
fit example figures, and `summary.json` with chosen settings and source checksums.
The saved records include placement for analysis; the counter does not receive it.

Verification completed:

- Every notebook code cell executed successfully; three counting figures and
  all comparison tables are embedded in its saved outputs.
- Recomputed all 24 candidate summaries and all 18 segmentation/counter
  summaries from the saved predictions; every MAE, within-two fraction, and
  bias matched. The lowest-fit-MAE settings match `count_reps` defaults.
- Every candidate has one nonnegative integer count for each of the 301
  training recordings. No prediction file includes a test recording.
- All example plots use fit recordings. The final test tables and later
  feature/prediction TODOs remain pending.
- A known 20-cycle sine signal is counted as 20 at both 50 Hz and 100 Hz.
  Flat signals and very short inputs return zero, and invalid signal choices
  raise a clear error. These check function behavior, not exercise accuracy.
- Source checksums match the saved settings; `util.py` and the data are
  unchanged. The execution helper remains deleted.

## Step 4: Window lengths and features

Status: complete. Length, features, and segmentation are fixed for the model
comparison; all current accuracy numbers are development results.

### 4.1 Plan before experiments

Use the supplied fixed Random Forest settings (`n_estimators=100`,
`random_state=42`) throughout this step. Train on fit windows and measure on
validation windows; do not tune the classifier yet or read test signals.

1. Compare 2-, 4-, and 6-second windows with adaptive segmentation and the seven
   baseline features. Select the window length using validation window accuracy.
2. At that length, compare the baseline with ten proposed additions: standard
   deviation of each axis, dominance of the strongest axis, strongest-axis
   dominant frequency and spectral concentration, three axis correlations, and
   acceleration RMS. Try meaningful groups and leave-one-feature-out comparisons.
3. With one chosen feature set and length, compare adaptive segmentation with
   the whole recording, baseline segmentation, and protocol trimming. Keep
   counting unchanged; choose the classification setup using development results.

We will document per-exercise changes instead of claiming that every feature
helps all exercises. Flat-window features must be finite. Feature extraction
must use only the window and sampling rate, never labels or recording identity.

### 4.2 Feature definitions and selection rule

Implemented ten candidate additions in `features.py`. Axis standard deviations
use the raw linear-acceleration axes; axis dominance is the largest of these
divided by their sum. Rhythm uses the strongest axis, a Hann taper, and FFT power
in 0.2–3 Hz. Spectral concentration is the largest frequency-bin power divided
by band power. Correlations describe coordinated axis motion; RMS measures total
acceleration intensity. Flat or undefined values become zero.

Selection is sequential: first choose the best baseline-feature window length
from 2/4/6 seconds, then compare feature groups and leave-one-feature-out sets
at that same length. Choose the highest validation window accuracy among sets
with 5–10 additions (the rubric's target), breaking exact ties by fewer features.
Single-group comparisons are diagnostic, not extra model/hyperparameter tuning.
Small differences remain uncertain because they all use one validation split.

Segmentation stays adaptive while choosing length/features. Afterwards, compare
classification under all four segmenters. Since the same segmentation also
serves counting, do not change it solely to chase a small classification gain;
interpret the classification and Step 3 counting results together.

### 4.3 Window comparison and feature-selection outcome

The baseline-feature window comparison gave:

| Window length | Validation window accuracy |
|---|---:|
| 2 seconds | 47.62% |
| 4 seconds | 48.93% |
| 6 seconds | 50.09% |

We selected six seconds for the feature comparison, following the declared
rule. Its gain over four seconds is only 1.16 percentage points, so this is a
practical choice rather than strong evidence that six seconds is generally best.
Longer windows can include more of a slow rep, but produce fewer windows and
coarser boundaries. Their frequency resolution is finer than shorter windows.

At six seconds with adaptive segmentation, the feature comparisons gave:

| Added feature group | Additions | Validation window accuracy |
|---|---:|---:|
| None | 0 | 50.09% |
| Axis motion | 4 | 52.09% |
| Rhythm | 2 | 50.09% |
| Correlations | 3 | 53.54% |
| RMS intensity | 1 | 49.00% |
| Axis motion + rhythm | 6 | 55.35% |
| Axis motion + correlations | 7 | 55.17% |
| Axis motion + rhythm + correlations | 9 | 53.18% |
| Axis motion + rhythm + RMS | 7 | 55.35% |
| All ten additions | 10 | 54.45% |
| **All except x–z correlation** | **9** | **56.26%** |

All ten leave-one-feature-out comparisons are saved. The selected set has the
highest validation accuracy among the declared eligible sets. It keeps all seven
baseline features and nine additions: `std_x`, `std_y`, `std_z`, `axis_dominance`,
`dominant_frequency`, `spectral_concentration`, `corr_xy`, `corr_yz`, and
`rms_acceleration`. Removing `corr_xz` improves this particular full-feature
forest by 1.81 points; it does not prove x–z coordination is physically useless.
The many subsets all share one small validation set, so tiny differences can
reflect selection noise. We will keep this fixed during model comparison.

### 4.4 Which exercises and features changed?

At six seconds, the selected features change validation **window recall** as
follows. Recall is the fraction of true windows of that exercise recognized
correctly. It is not overall accuracy or recording-level recall.

| Exercise | Validation windows | Baseline recall | Selected recall | Change (percentage points) |
|---|---:|---:|---:|---:|
| Ab crunches | 38 | 2.63% | 2.63% | 0.00 |
| Bicep curls | 76 | 44.74% | 43.42% | -1.32 |
| Jumping jacks | 92 | 93.48% | 93.48% | 0.00 |
| Lat raises | 25 | 20.00% | 28.00% | +8.00 |
| Lunges | 63 | 30.16% | 55.56% | +25.40 |
| Push-ups | 49 | 6.12% | 53.06% | +46.94 |
| Sit-ups | 37 | 21.62% | 21.62% | 0.00 |
| Squats | 171 | 70.18% | 66.67% | -3.51 |

The overall gain is concentrated in push-ups and lunges; the features do not
help all classes equally. Jumping jacks already have high recall. Ab crunches
remain very difficult. Lat raises improve on only three validation recordings,
so the apparent gain is uncertain.

On identical six-second adaptive windows, push-ups predicted as lunges fall
from **19 to 6**, and correctly predicted push-up windows rise from **3 to 26**.
Lunges predicted as push-ups fall from **9 to 3**. The axis-motion group alone
reduces push-ups→lunges from 19 to 8; the correlation group alone reduces
push-ups→squats from 19 to 14. These are concrete exercise-pair examples for the
report's feature table, with all model settings held fixed.

The group comparisons provide more specific evidence:

- Axis-motion additions raise push-up recall by 28.57 points, lunges by 20.63,
  and lat raises by 12.00, while squat recall falls by 11.70. Which axes move can
  help distinguish movement patterns, but phone orientation/placement also varies.
- Rhythm additions alone leave overall accuracy unchanged. Push-up recall rises
  by 28.57 points while bicep-curl recall falls by 10.53 and some others decline.
  A dominant signal frequency can be a harmonic rather than the true rep rate.
- Correlations alone raise overall accuracy by 3.45 points and push-up recall
  by 32.65. Coordinated motion between axes adds information beyond magnitude.
- RMS alone lowers overall accuracy by 1.09 points. Intensity overlaps with the
  baseline magnitude statistics, although it may still interact with other inputs.

The physical interpretations are hypotheses consistent with the signal design;
the observed accuracy/recall differences are what we actually measured.

For the report's feature table, leave-one-out effects below are **conditional
on all other additions being present**: full-feature accuracy minus accuracy
with that feature removed. Positive means adding it back improved this forest.

| Feature | Conditional accuracy effect (points) | Rationale |
|---|---:|---|
| `std_x` | -0.73 | Motion amplitude along x |
| `std_y` | +1.45 | Motion amplitude along y |
| `std_z` | -0.73 | Motion amplitude along z |
| `axis_dominance` | 0.00 | Concentration of motion along one axis |
| `dominant_frequency` | -1.63 | Movement rhythm, including possible harmonics |
| `spectral_concentration` | -0.54 | Regularity of the dominant frequency |
| `corr_xy` | +1.81 | Coordination of x and y motion |
| `corr_xz` (dropped) | -1.81 | Coordination of x and z motion |
| `corr_yz` | -0.91 | Coordination of y and z motion |
| `rms_acceleration` | +1.27 | Total acceleration intensity |

These small conditional effects do not establish universal feature importance.
For example, RMS hurts alone but helps in the full set; rhythm helps with axis
features despite no gain alone. Dropping one input also changes the forest's
random feature subsets, so these are practical comparisons, not isolated causal
effects. The final selection follows the declared whole-set accuracy rule.

### 4.5 Segmentation check and locked setup

Using the selected 16 features and six-second windows:

| Segmentation | Fit windows | Validation windows | Validation window accuracy |
|---|---:|---:|---:|
| Adaptive | 1,674 | 551 | 56.26% |
| Whole recording | 2,570 | 801 | 53.43% |
| Supplied baseline | 2,331 | 731 | 53.35% |
| Protocol trim | 1,817 | 550 | 54.91% |

Adaptive is highest on this split and also useful for counting in Step 3, so
keep it as the shared method. Its gain over whole-recording classification is
2.83 points; over protocol trimming it is only 1.35 points. Since segmentation
changes which windows are evaluated, these are practical pipeline comparisons
on the same recordings rather than identical-window comparisons.

Locked development setup for Step 5:

- Adaptive segmentation with unchanged Step 2 settings.
- Six-second windows with the supplied 50% overlap.
- Seven baseline features plus the nine selected additions (16 total).
- The Step 3 PCA rep counter, unchanged.
- No classifier hyperparameter tuning yet; Step 4 used the same fixed forest.

The module's `add_my_features` now returns exactly these 16 columns. The notebook
sets `WINDOW_SEC`, `FEATURES`, `frames`, `train_frames`, `fit_frames`, and
`val_frames` for the next sections. Test frames are deliberately not built yet.

### 4.6 Saved outputs and verification

Run `.venv/bin/python A3-ExerciseRecognition/feature_exploration.py` from the
repository root to reproduce the comparisons. `results/step4_features/` contains
all feature frames, per-run validation predictions/reports, group/leave-one-out
tables, fit feature medians, the selected training frame table, a per-exercise
recall figure, and the exact selected settings/source checksums.

Checks already completed: synthetic signals at 50 and 100 Hz recover the known
dominant frequency, correlations, axis dominance, and RMS. The original seven
features are preserved exactly for ordinary windows. Flat and one-sample
windows produce finite values.

Final verification completed:

- Every notebook code cell ran successfully, with the comparison tables and
  per-exercise recall figure saved in notebook outputs.
- Recomputed all 27 validation accuracies and per-class reports from saved
  predictions; all matched. Every comparison covers all 76 validation recordings.
- All feature files contain only the 301 training recordings, with finite
  features and no test recordings.
- The selected length/features match the declared selection rule. Directly
  extracting a selected window reproduces the saved 16 feature values.
- Source checksums match. `util.py`, the segmentation method, and the Step 3
  counter are unchanged. The execution helper stays deleted.
- Classifier tuning, recording-level prediction, and final test tables remain
  pending; they were not silently completed using validation results.

## Step 5: Tune and compare classifiers

Status: complete. Model choices are frozen and the final test-window results
are reported. Recording-level evaluation remains Step 6.

### 5.1 Plan before model tuning or test evaluation

Keep Step 4's adaptive segmentation, six-second windows, and 16 features fixed.
Fit candidates on the 225 fit recordings (1,674 windows); select settings by
window accuracy on 76 validation recordings (551 windows).

Declared grids:

- Decision Tree: `max_depth=[3,5,8,12,None]`, `criterion=['gini','entropy']`,
  and `class_weight=[None,'balanced']` (20 combinations).
- Random Forest: `n_estimators=[100,200]`, `max_depth=[None,8,16]`,
  `min_samples_leaf=[1,3]`, and `class_weight=[None,'balanced']` (24 combinations).
- Both use `random_state=42`; other settings stay at library defaults.
- Pick the highest validation window accuracy; resolve exact ties by the first
  combination in the declared grid order. Choose the final model family by
  validation accuracy as well, before examining test results.

Compare class balancing at matched settings using validation data, and explain
whether it helps the smaller classes. The provided default Decision Tree
(depth 5, entropy) and fixed 100-tree forest remain reference models.

After selection, save the frozen choices, retrain default and selected models
on all 301 training recordings (2,225 windows), and only then build/evaluate test
windows. Save predictions, classification reports, test-window confusion
matrices, the supplied Decision Tree visualization, and forest importances.
Do not revise settings or feature choices after seeing test performance.
Recording-level voting stays in Step 6.

### 5.2 Validation choices frozen before test access

The declared grid search selected:

- **Decision Tree:** `max_depth=12`, `criterion='gini'`, `class_weight=None`.
  Validation accuracy is **50.09%**, versus **43.74%** for the provided default
  tree (depth 5, entropy, no class weighting).
- **Random Forest:** `n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`,
  `class_weight='balanced'`. Validation accuracy is **56.62%**, versus **56.26%**
  for the fixed 100-tree forest from Step 4.

The Random Forest family was selected by validation accuracy before loading test
signals. The choices and input/source checksums are saved in
`results/step5_classification/frozen_selection.json`. Subsequent notebook runs
reuse those choices and the completed evaluation; they do not select new
settings from test performance. Changes to frozen inputs trigger a review error.

The tuned forest's gain over the default is only **two of 551 validation
windows** (0.36 points). Treat this as a modest settings choice, not evidence of
a substantial tuning improvement. Both classifiers use exactly the same windows
and features, so their model comparison is controlled.

### 5.3 Class balancing at matched settings

We compared `class_weight=None` with `'balanced'` while keeping the selected
model's other settings identical, using validation data only.

- For the 200-tree forest, accuracy is **56.44% without weighting** and **56.62%
  with balancing**: only one validation window of overall difference. Balancing
  changes recall: ab crunches 2.63%→5.26%, lat raises 32%→36%, sit-ups
  18.92%→35.14%, and push-ups 51.02%→61.22%. Squat recall falls
  67.25%→57.89%. This redistributes performance away from the largest class;
  it does not make all classes better or solve ab crunches.
- For the depth-12 Gini tree, balancing lowers accuracy from **50.09% to
  48.64%**. Some recalls increase (e.g., lunges 47.62%→61.90%), while lat raises
  falls 28%→20% and squats 56.14%→45.03%. We retain the unweighted tree according
  to the predeclared accuracy rule.

Per-exercise values and all 44 grid results are saved. Small classes and
overlapping windows make small validation changes especially uncertain.
The balancing weights are calculated from training **windows**, not recording
counts; longer recordings also influence each class's effective representation.

### 5.4 First final test-window evaluation

After freezing choices, retrained all four reference/selected models on **2,225
windows from all 301 training recordings**. Only then built **925 windows from
129 test recordings**, using the unchanged six-second/16-feature/adaptive setup.

| Model | Settings | Validation window accuracy | Test window accuracy | Test macro F1 |
|---|---|---:|---:|---:|
| Default Decision Tree | depth 5, entropy, unweighted | 43.74% | 48.43% | 0.280 |
| Selected Decision Tree | depth 12, Gini, unweighted | 50.09% | 46.49% | 0.406 |
| Default Random Forest | 100 trees, unweighted | 56.26% | 59.57% | 0.493 |
| Selected Random Forest | 200 trees, balanced | 56.62% | 60.43% | 0.553 |

The validation-selected tree has **lower test accuracy** than the default.
We did not change it after seeing that result. Its macro F1 is higher, showing
that the accuracy change alone does not describe its distribution of errors.
Macro F1 averages class F1 values equally, while accuracy counts every window.

The selected forest is **13.95 percentage points** higher in test-window accuracy
than the selected tree, but only **0.86 points** higher than the default forest.
The larger observed gain comes from classifier type rather than the forest
tuning. The selected model family remains the validation-chosen Random Forest.
These are window results; a real app's recording-level results are still Step 6.

### 5.5 Which exercises benefit from the model change?

Test-window recall for each selected model:

| Exercise | Decision Tree | Random Forest | Difference (points) |
|---|---:|---:|---:|
| Ab crunches | 12.82% | 33.33% | +20.51 |
| Bicep curls | 37.90% | 45.16% | +7.26 |
| Jumping jacks | 88.73% | 98.59% | +9.86 |
| Lat raises | 33.33% | 58.97% | +25.64 |
| Lunges | 51.02% | 60.20% | +9.18 |
| Push-ups | 34.21% | 55.26% | +21.05 |
| Sit-ups | 21.57% | 42.16% | +20.59 |
| Squats | 46.23% | 60.00% | +13.77 |

Recall improves for every exercise here, but by different amounts. Lat raises,
push-ups, sit-ups, and ab crunches gain more than bicep curls. Jumping jacks are
easy for both models, while ab crunches still have low recall. The forest
combines many trees rather than depending on one tree's thresholds; that is a
plausible reason for improvement, not proof of the cause of each error.

Remaining examples in the forest's window confusion matrix: ab crunches→squats
18 windows, lunges→squats 22, squats→sit-ups 39, and squats→lunges 32. These are
descriptive final test errors to investigate later; they will not be used to
retune the methods. Recording-level confusions may differ after voting.

### 5.6 Figures, models, and notebook integration

`classification.py` handles the declared grids and deterministic construction.
`classification_exploration.py` freezes validation choices, retrains models,
and saves their test predictions/reports, confusion matrices, tree view, and
forest feature importances under `results/step5_classification/`.

The tree uses the supplied `viz_tree` helper unchanged. The notebook displays
its top two levels for readability. Forest importances are impurity-based and
can share credit among correlated inputs; they are not causal evidence. Its
largest entries include `std_y`, RMS, and dominant frequency.

Saved `.joblib` models allow Step 6 to use exactly the classifiers evaluated
here. The notebook's two classifier sections will display the completed tables
and figures and expose `best_dt_model`, `best_rf_model`, and `test_frames`.
Recording voting and rep-counter final-test comparisons remain pending.

Verification completed:

- Recomputed all 44 validation candidate accuracies and macro F1 scores from
  saved predictions. Both selected candidates follow the declared accuracy rule.
- The selected model family matches validation selection, not a test comparison.
- All four saved models reproduce their saved test predictions exactly. Their
  feature names/order and hyperparameters match the frozen configuration.
- Every validation candidate covers the 76 validation recordings and excludes
  test recordings. Test windows cover exactly 129 recordings and exclude all
  training recordings. All 16 feature columns are finite.
- All notebook code cells executed successfully. Both classifier tables and
  four model figures are embedded. The tree's top-two-level view is readable.
- The notebook explicitly reused the frozen choices and cached test evaluation,
  rather than reselecting settings after test results were known.
- Source/configuration checksums match and `util.py` is unchanged. Recording
  voting TODOs remain for Step 6, and the execution helper stays deleted.

## Step 6: Recording-level predictions

Status: complete. Final recording-level predictions and confusion matrices are
saved; physical error attribution remains Step 7.

### 6.1 Plan before recording-level evaluation

Use the saved validation-selected Decision Tree and Random Forest from Step 5.
Do not retrain, tune, or switch the selected model family from recording-level
test performance. The final family remains the validation-chosen Random Forest.

Implement the required `predict_recordings(model, frames, features=None)` in a
separate module and call it from notebook Function 4. Each window gets one vote;
the most common label wins. Resolve exact ties alphabetically, as a fixed and
reproducible rule. Do not try confidence weighting or tune a tie rule on test.
The function returns exactly `recording`, `true`, and `predicted` columns.

Evaluate one answer for each of the 129 test recordings. Save both models'
recording-level predictions, per-exercise reports, and 8×8 confusion matrices.
Compare window accuracy with recording accuracy and inspect vote disagreements.
Also calculate mean within-recording window accuracy so that both comparisons
give recordings equal weight: ordinary window accuracy gives long recordings
more influence. Keep diagnostics separate from the required three-column output.

### 6.2 Results with the fixed vote rule

Both selected models use the same 925 test windows, covering 129 recordings.
No fitting or parameter search occurred in this step.

| Model | Window accuracy | Mean within-recording window accuracy | Recording accuracy | Correct recordings | Recording macro F1 |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 46.49% | 47.53% | 48.84% | 63/129 | 0.402 |
| Random Forest | 60.43% | 59.04% | 63.57% | 82/129 | 0.591 |

Mean within-recording window accuracy averages each recording's fraction of
correct windows, giving each recording equal weight. Recording accuracy gives
each recording one correct/incorrect answer after voting. Voting is higher
than this equal-weight reference by 1.31 points for the tree and 4.53 for the
forest. The overall difference is not just a consequence of weighting lengths
differently. The final family remains the validation-selected Random Forest.

### 6.3 Which recordings/exercises benefit?

For the forest, compare mean within-recording window accuracy with voting recall
within each exercise. These describe the same recordings with equal weights:

| Exercise | Test recordings | Mean window accuracy per recording | Recording recall |
|---|---:|---:|---:|
| Ab crunches | 8 | 15.97% | 25.00% (2/8) |
| Bicep curls | 15 | 46.22% | 53.33% (8/15) |
| Jumping jacks | 26 | 98.90% | 100.00% (26/26) |
| Lat raises | 6 | 43.35% | 66.67% (4/6) |
| Lunges | 13 | 65.10% | 76.92% (10/13) |
| Push-ups | 17 | 48.98% | 58.82% (10/17) |
| Sit-ups | 12 | 44.38% | 41.67% (5/12) |
| Squats | 32 | 54.74% | 53.12% (17/32) |

Voting helps most exercise groups on this test set, but slightly hurts sit-ups
and squats compared with the equal-weight window reference. It cannot fix
consistent mistakes across windows. The small number of recordings in some
classes makes their percentages uncertain.

The forest has **11 tied recording votes**, four resolved correctly; the tree
has **15 ties**, four correct. Alphabetical tie-breaking is reproducible but
arbitrary. We report its consequences instead of selecting a better tie rule
using test labels.

Concrete limitations:

- Sit-ups `024_2026-10-18_23-55-25.csv`: two forest windows split between sit-ups
  and ab crunches. Alphabetical tie-breaking gives ab crunches, despite one
  correct window.
- Squats `024_2026-10-14_22-16-05.csv`: three of six windows say squats and three
  say sit-ups. The tie resolves to sit-ups.
- A vote can also be correct with less than half the windows: sit-ups
  `033_2026-10-19_02-57-06.csv` has four correct windows of nine, with other
  labels split more widely. The most common label wins; a strict majority above
  50% is not required by this usual multiclass voting rule.

### 6.4 Final recording-level confusion matrix

The final forest matrix is saved at
`results/step6_recordings/figures/random_forest_recording_confusion.png` and
embedded in the notebook. It is 8×8, has a total of 129, and has 82 diagonal
entries in total. Rows are true exercises, columns are predictions.

Some larger off-diagonal cells, counting recordings:

- Squats→push-ups: 4; squats→sit-ups: 4; squats→lunges: 3.
- Ab crunches→push-ups: 3; ab crunches→squats: 3.
- Bicep curls→squats: 3; sit-ups→ab crunches: 3.

These are descriptive final errors. Their segmentation/feature/model causes
will be investigated with traces and feature values in Step 7, rather than
guessed from the matrix alone. Saved errors already include placement and
reported reps for that later analysis; those fields do not affect predictions.

### 6.5 Implementation and outputs

`predict_recordings` is in `recording_predictions.py` and is called by notebook
Function 4. It predicts windows using only the feature columns, groups by
recording, and returns exactly `recording`, `true`, and `predicted`. The true
column is copied for evaluation, not used to choose the vote. Inconsistent
true labels within a recording are rejected; empty inputs return an empty
three-column table.

`recording_exploration.py` loads the frozen Step 5 models and checks their window
predictions still match Step 5. It writes predictions, separate vote diagnostics,
classification reports, comparison and per-exercise tables, and both confusion
matrices under `results/step6_recordings/`. It also saves
`final_recording_predictions.csv` and `final_classification_errors.csv` for the
report and Step 7. No models are retrained or selected here.

Reproduce from the repository root:

```bash
.venv/bin/python A3-ExerciseRecognition/recording_exploration.py
```

The notebook now exposes `dt_recordings`, `rf_recordings`, and `recordings`
(the fixed final forest's predictions), and contains the completed table and
interpretation.

### 6.6 Verification completed

- All notebook code cells ran successfully. Function 4 and the recording section
  are completed, with both 8×8 matrices embedded. Step 7 TODOs remain.
- Each model returns exactly 129 unique recordings, with correct true labels,
  and no training recording appears in test predictions.
- Recomputed both confusion matrices, accuracies, macro F1 values, and vote
  diagnostics from predictions. Matrix totals are 129; diagonal sums are 63
  and 82. The final file is exactly the forest predictions, with 47 errors.
- Direct predictions from the saved models match the function and saved files;
  model hashes and Step 5's frozen selection are unchanged.
- Synthetic vote checks cover majority, alphabetical ties, single windows,
  explicit/default feature lists, row-order independence, empty frames, and
  rejection of inconsistent true labels or missing recording identifiers.
- Source hashes match, `util.py` is unchanged, and no new model fitting or
  parameter selection occurred. The execution helper stays deleted.

## Next step and final deliverables

Next is **Step 7: failure analysis and remaining final counting comparisons**.
Investigate the recording-level confusions using traces and feature values,
attribute meaningful errors to segmentation/features/model behavior, and choose
two clear failure modes for the report. Evaluate the frozen counter/segmentation
alternatives on test recordings for the remaining required tables. Do not tune
the methods from those results.

Steps 1–4 contain development measurements. Steps 5–6 contain final classifier
test results at window and recording levels. The final repetition/segmentation
test tables, detailed failure attribution, and `exploration_report.pdf` remain
unfinished. Test errors may be inspected and explained, but must not drive
further tuning of methods or settings.

Submission requires the completed notebook, needed Python files, and report in
one ZIP. The report sections are segmentation, classification, counting, final
pipeline and error attribution, top two failures, and contributions.
