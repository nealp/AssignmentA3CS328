# Rubric checklist and personal verification

This checklist follows the supplied `RUBRIC.md` and starter notebook. It records
our current evidence; it does not predict a grade. The rubric gives category
totals, not a point allocation for each function or a minimum accuracy target.

## Current readiness

The pipeline and classifier evaluations are implemented. The assignment is
**not ready for submission yet**: final counting tables, detailed failure
analysis, the PDF report, and contributions still need completion.

| Graded category | Points | What we can currently verify |
|---|---:|---|
| Group code | 12 | Main methods and classifier evaluations implemented; remaining final counting/failure-analysis sections incomplete |
| Group exploration report + contributions | 12 | Evidence in notebook/log; required PDF and contribution statement not yet present |
| Individual assignment quiz | 12 | You must understand your group's actual choices/results; cannot be verified from the repository |
| Individual data collection | 12 | Check your own course record; cannot be verified from this anonymized corpus |

The general concept quiz is graded separately. AI is permitted for code/report;
the individual assignment quiz is closed-resource and refers to your report.

## Code and experiment requirements

- [x] Use the supplied corpus and manifest; all 430 recordings are present.
- [x] Keep `util.py` unchanged.
- [x] Use the prescribed recording split: 301 train / 129 test, with 225 fit /
  76 validation, `random_state=42`. No recording crosses splits.
- [x] Select methods/settings using training/validation, then freeze before test.
- [x] Implement Function 1: `find_exercise_segment`.
- [x] Compare baseline segmentation with at least one original approach;
  adaptive and protocol trimming were compared, including effects on counting
  and classification.
- [x] Plot at least six fit recordings across exercises and placements;
  ten were inspected, plus explicit baseline/adaptive failure examples.
- [x] Implement Function 2: `count_reps`; compare candidate signals, filtering,
  peak distance, and prominence on development data.
- [ ] Fill all five required **test** segmentation/counter combination rows.
- [ ] Fill final **test** counting breakdowns by placement, and inspect errors
  by exercise. Development results must not be substituted for test results.
- [ ] Evaluate each tried segmentation with the final counter on test for report §1.
- [ ] Evaluate the tried rep counters with final segmentation on test for report §3.
- [x] Implement Function 3: `add_my_features`; seven baseline plus nine selected
  additions, with finite values for flat/short windows.
- [x] Compare at least three window lengths: 2, 4, and 6 seconds.
- [x] Compare baseline/new features, additions/removals, and segmentation versus
  no segmentation using the same fixed forest during feature exploration.
- [x] Fix one length, feature set, and segmentation before comparing models:
  6 seconds, 16 features, adaptive segmentation.
- [x] Tune Decision Tree depth/criterion on fit/validation; also try balancing.
- [x] Retrain default and validation-selected tree on all training recordings;
  report both test results and show the tree/test-window matrix.
- [x] Tune a different classifier type on the same features/windows: Random Forest.
- [x] Show forest test-window matrix, feature importances, and comparison by class.
- [x] Implement Function 4: `predict_recordings`, with one row per recording and
  exactly `recording`, `true`, `predicted` columns.
- [x] Compare both selected models' window/recording accuracy and discuss voting.
- [x] Show the final 8×8 **recording-level** test confusion matrix.
- [ ] Attribute every meaningfully nonzero confusion cell to segmentation,
  feature limitations, or model behavior, supported by evidence.
- [ ] Complete the top-two failure analysis in the notebook, including required
  traces/peaks and feature comparisons for classification failures.
- [ ] Remove all remaining `FILL UP`, unfinished TODOs, and `Write Below` prompts.

## PDF report: six required sections

- [ ] **§1 Segmentation:** test comparison table using final counter; exact
  parameters; which approach won/failed and why; classification effect; one
  baseline-failure figure.
- [ ] **§2 Classification:** tried/kept/dropped features, exercises separated,
  measured effects and rationale; model parameters and both window/recording
  accuracies; explain unequal gains across exercises.
- [ ] **§3 Counting:** test comparison of tried counters with final segmentation;
  signals/settings, MAE, within ±2; explain hardest exercises/placements.
- [ ] **§4 Final pipeline:** exact segmentation/window/features/model/counter;
  final recording matrix; explain and attribute meaningful off-diagonal cells.
- [ ] **§5 Two failures:** for each, identify a specific recording, include its
  trace/segment and detected peaks where relevant, explain the physical
  mechanism, name the responsible stage, and suggest one concrete improvement.
- [ ] **§6 Contributions:** honestly name who did what; do not invent contributions.
- [ ] Every number agrees with saved predictions and notebook results, with
  validation/test and window/recording units clearly labeled.
- [ ] Figures are readable and embedded in `exploration_report.pdf`; paths alone
  are insufficient. Include sample sizes and small-class limitations.

## What has been checked technically?

- Saved notebook code cells executed without errors; unfinished analysis cells
  can contain only comments, so this alone does not establish completion.
- Saved prediction files reproduce reported metrics; confusion matrices have
  correct totals. Final classifier recording matrix totals 129, diagonal 82.
- Synthetic checks cover the counting signal, feature definitions, voting ties,
  empty inputs, and related edge cases at 50/100 Hz where applicable.
- Frozen model settings and saved model/source hashes were checked.
- Clean-copy execution without saved results: passed. Using the existing
  installed environment, all 17 code cells ran in a temporary copy starting
  with no `results/` folder. It regenerated artifacts and reproduced the same
  selected model settings, test-window scores, and final recording predictions.
  Original data was accessed read-only; the source notebook was not modified.
  This verifies regeneration in our environment, not every possible machine.
  Empty analysis cells still ran without completing their requirements.

## How you should check it yourself

1. Finish the unchecked requirements above before treating this as final.
2. Open `A3_exercise_recognition.ipynb` in Jupyter with the project environment.
   Use **Restart Kernel and Run All Cells**, then save it. Read the output of
   every experiment, including any errors or warnings.
3. Search the notebook for `TODO`, `FILL UP`, and `Write Below`. Inspect each
   occurrence; completed explanatory instructions are fine, empty work is not.
4. Match each results-table value to the corresponding CSV/JSON under `results/`.
   Verify which split and unit it describes. For a confusion matrix, rows are
   true labels and columns predicted labels; its sum equals the evaluated count.
5. Read the final PDF beside this checklist and tick each required report item.
6. Unzip the submission into a separate folder and run it using the required data
   and dependencies. Include every Python module imported by the notebook.
   Ensure cached artifacts are included when relied on, or that the notebook can
   regenerate them. Do not depend on a machine-specific `/private/tmp` backup.
7. Open the ZIP and PDF once more; verify one group submission is uploaded and
   accepted by Canvas. A GitHub push is not a Canvas submission.

Setup from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r A3-ExerciseRecognition/requirements.txt
.venv/bin/jupyter lab --notebook-dir=A3-ExerciseRecognition
```

## Your own understanding: practice before the closed-resource quiz

Explain these in your own words without looking at the code, then check your
answer against the actual code/report. These are study prompts, not known quiz
questions or a substitute for your course's rules.

1. What does each stage of our pipeline do? Where can an early error propagate?
2. Why split recordings before windowing? What are fit, validation, and test for?
3. How does our adaptive segmenter work, including threshold, gaps, bouts, padding,
   and fallback? Show a case where it cuts valid activity.
4. Why can magnitude count two peaks per rep? What does PCA change? Why can a
   1 Hz filter miss fast movement, and why can prominence miss weak reps?
5. What do MAE, within ±2, and bias measure? Can bias be near zero with large errors?
6. Which nine features did we keep, why were they plausible, and which one was
   dropped? Why is a small validation difference uncertain?
7. Why six-second windows? Why does a frequency peak not necessarily equal rep rate?
8. What settings did the two selected models use, and how were they selected?
9. What does balanced class weighting change? Why did it help some recalls and
   hurt squat recall? Are weights based on windows or recording counts?
10. Why did the selected tree score worse than default on test accuracy? What
    does the macro F1 comparison show, and why didn't we retune on test?
11. How does voting work, including ties? Why are 925-window accuracy and
    129-recording accuracy different? When can voting make a result worse?
12. Explain both final failure examples using their actual traces, mechanisms,
    responsible stages, and suggested improvements. This remains pending Step 7.
13. Describe your own contribution and interpret your group's report, including
    limitations and unsuccessful experiments, without relying on AI assistance.

Numbers to distinguish: classifier test recording accuracy is **63.57% (82/129)**.
The counter's **3.33-rep MAE** is currently a **validation** result; its final test
table is still pending. Do not mix those two evaluation claims.

## Personal administrative checks

- [ ] Confirm your own 12-point data-collection requirement and recorded credit.
- [ ] Confirm the assignment quiz date/rules and your submission/attendance.
- [ ] Confirm group membership and actual contribution statement.
- [ ] Verify the applicable deadline/extensions and remaining shared late days
  in Canvas or with course staff. The repo lists Nov 1 at 11:59 PM, five shared
  late days across assignments, and no quiz late days except emergencies.

Full marks cannot be guaranteed by passing runtime checks. Correct experiments,
complete evidence and explanations, individual understanding, data-collection
credit, and applicable submission policies all matter.
