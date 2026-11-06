"""Final descriptive test evaluation of already frozen counting alternatives.

Run from this directory. Does not fit or change any pipeline settings.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from baseline import count_summary
from util import prepare_recording, cut_segment, baseline_segment, no_segment, baseline_count_reps
from segmentation import adaptive_segment, protocol_segment
from rep_counting import REP_SETTINGS, rep_details


def main():
    out = Path('results/report_evidence')
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv('results/step1_baseline/test_manifest.csv')
    settings = json.loads(Path('results/step3_rep_counting/summary.json').read_text())['candidate_settings']
    segments = dict(none=no_segment, baseline=baseline_segment, adaptive=adaptive_segment,
                    protocol_trim=protocol_segment)
    rows, diagnostics = [], []
    for row in manifest.itertuples():
        data, rate = prepare_recording(row.path)
        bounds = {name: fn(data, rate) for name, fn in segments.items()}
        for seg_name, bounds_pair in bounds.items():
            part = cut_segment(data, *bounds_pair)
            candidates = settings if seg_name == 'adaptive' else {'selected': REP_SETTINGS, 'baseline': None}
            for name, config in candidates.items():
                count = baseline_count_reps(part, rate) if config is None else len(rep_details(part, rate, **config)['peaks'])
                rows.append(dict(recording=row.path, exercise=row.exercise, placement=row.placement,
                    true_reps=row.reps, pred_reps=count, error=count-row.reps,
                    segmentation=seg_name, counter=name))
        diagnostics.append(dict(recording=row.path, duration=float(data.time.iloc[-1]),
            segment_start=bounds['adaptive'][0], segment_end=bounds['adaptive'][1],
            axis_std_max=float(data[['x','y','z']].std().max())))
    values = pd.DataFrame(rows)
    values.to_csv(out/'count_predictions_test.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out/'signal_diagnostics_test.csv', index=False)
    summary = []
    for (seg, counter), part in values.groupby(['segmentation','counter'], sort=False):
        summary.append(dict(segmentation=seg, counter=counter, **count_summary(part)))
    pd.DataFrame(summary).to_csv(out/'count_comparison_test.csv', index=False)
    selected_name = next(name for name, config in settings.items() if config == REP_SETTINGS)
    final = values[(values.segmentation == 'adaptive') & (values.counter == selected_name)]
    for group in ('exercise','placement'):
        count_summary(final, by=group).to_csv(out/f'count_test_by_{group}.csv')
    # One clear signal failure and one systematic classification confusion.
    failures = pd.read_csv('results/step6_recordings/final_classification_errors.csv')
    chosen_class = failures[(failures.true == 'Squats') & (failures.predicted == 'Sit-ups')].iloc[0]
    chosen_count = final.loc[final.error.abs().idxmax()]
    examples = []
    for label, row in [('classification', chosen_class), ('counting', chosen_count)]:
        data, rate = prepare_recording(row.recording)
        start, end = adaptive_segment(data, rate)
        part = cut_segment(data,start,end)
        detail = rep_details(part,rate,**REP_SETTINGS)
        fig, axes = plt.subplots(2,1,figsize=(10,4.8))
        axes[0].plot(data.time,data.filtered_accel_mag,lw=.7,color='#425466')
        axes[0].axvspan(start,end,alpha=.18,color='#238b8e',label='Selected segment')
        axes[0].set_ylabel('Magnitude (m/s²)'); axes[0].legend(loc='upper right')
        axes[1].plot(part.time,detail['signal'],lw=.7,color='#238b8e')
        peaks=detail['peaks']
        axes[1].plot(part.time.iloc[peaks],detail['signal'][peaks],'rx',ms=4,label=f'{len(peaks)} detected peaks')
        axes[1].set_ylabel('PCA signal (m/s²)'); axes[1].set_xlabel('Time (s)'); axes[1].legend()
        fig.suptitle(row.recording+' | '+row.placement,fontsize=9)
        fig.tight_layout(); fig.savefig(out/f'failure_{label}.png',dpi=160); plt.close(fig)
        examples.append(dict(kind=label, **row.to_dict(), segment_start=start, segment_end=end,
                             counted=len(peaks), full_axis_std=data[['x','y','z']].std().to_dict()))
    (out/'examples.json').write_text(json.dumps(examples,indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print(json.dumps(examples,indent=2))


if __name__ == '__main__':
    main()
