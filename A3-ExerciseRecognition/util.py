# -- IMPORTS START --
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt, find_peaks
# -- IMPORTS END --

# enable zooming into graphs
plt.rcParams['figure.figsize'] = [9, 6] # width, height in inches


# Helper function to visualize model - Do not modify
def viz_tree(dt_model,features_frames,cnames,max_depth=3):
    # Fix feature names as list
    feature_names = features_frames.columns.tolist()

    # Only the top `max_depth` levels are drawn so the boxes stay readable
    fig, ax = plt.subplots(figsize=(20,8))
    tree.plot_tree(dt_model,
                   feature_names=feature_names,
                   fontsize=8,
                   class_names=cnames,
                   filled=True,
                   max_depth=max_depth,
                   ax=ax)

    plt.title('Decision Tree')
    plt.savefig('dt.png')


#Do not modify
def load_manifest(root='./data'):
    # One row per recording: path, exercise, reps, placement
    return pd.read_csv(os.path.join(root, 'manifest.csv'))


#Do not modify
def split_recordings(manifest, test_size=0.3, random_state=42):
    # Split whole recordings (not windows), with the same mix of exercises in both parts.
    # Windows of one recording are very similar, so they must never be in both train and test.
    train, test = train_test_split(manifest, test_size=test_size, random_state=random_state,
                                   stratify=manifest['exercise'])
    return train.reset_index(drop=True), test.reset_index(drop=True)


#Do not modify
def load_recording(path):
    # Sensor Logger accelerometer CSV -> time in seconds from the start, and the x, y, z axes (m/s^2)
    raw = pd.read_csv(path)
    data = pd.DataFrame({'time': raw['seconds_elapsed'] - raw['seconds_elapsed'].iloc[0],
                         'x': raw['x'], 'y': raw['y'], 'z': raw['z']})

    # Most phones log at ~100 Hz, a few at 50-60 Hz, so estimate it from the timestamps
    sample_rate = 1 / data['time'].diff().median()
    return data, sample_rate


#Do not modify
def calc_magnitude(data):

    # Calculate magnitude
    data['accel_mag'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2) # absolute accel magnitude
    data['accel_mag'] = data['accel_mag'] - data['accel_mag'].mean() # detrend: "remove gravity"

    return data


#Do not modify
def remove_noise(data,sampling_rate):

    # Low pass filter
    cutoff = 5 # Hz
    order = 2
    b, a = butter(order, cutoff/(sampling_rate/2), btype='lowpass')
    data['filtered_accel_mag'] = filtfilt(b, a, data['accel_mag'])

    return data


#Do not modify
def prepare_recording(path):
    # load -> magnitude -> remove noise
    # Columns of data: time, x, y, z, accel_mag, filtered_accel_mag
    data, sample_rate = load_recording(path)
    data = calc_magnitude(data)
    data = remove_noise(data, sample_rate)
    return data, sample_rate


#Do not modify
def add_features(window):
    features = {}
    features['avg'] = window['filtered_accel_mag'].mean()
    features['max'] = window['filtered_accel_mag'].quantile(1)
    features['med'] = window['filtered_accel_mag'].quantile(0.5)
    features['min'] = window['filtered_accel_mag'].quantile(0)
    features['q25'] = window['filtered_accel_mag'].quantile(0.25)
    features['q75'] = window['filtered_accel_mag'].quantile(0.75)
    features['std'] = window['filtered_accel_mag'].std()
    df = pd.DataFrame([features])
    return df


#Do not modify
def baseline_features(window, sample_rate):
    # The A2 baseline features, in the (window, sample_rate) form that build_feature_frames expects
    return add_features(window)


#Do not modify
def no_segment(data, sample_rate):
    # Use the whole recording
    return data['time'].iloc[0], data['time'].iloc[-1]


#Do not modify
def baseline_segment(data, sample_rate, threshold=0.5):
    # Moving standard deviation of the filtered magnitude over 1 s
    win = max(int(sample_rate), 2)
    moving_std = data['filtered_accel_mag'].rolling(win, center=True, min_periods=1).std()

    # Active = above a fixed threshold (m/s^2). The segment goes from the first to the last active sample.
    active = data['time'][moving_std > threshold]
    if len(active) == 0:
        return no_segment(data, sample_rate)
    return active.iloc[0], active.iloc[-1]


#Do not modify
def cut_segment(data, start_sec, end_sec):
    # Keep only the samples between start_sec and end_sec
    segment = data[(data['time'] >= start_sec) & (data['time'] <= end_sec)].reset_index(drop=True)

    # A segment shorter than 1 s cannot be used, so fall back to the whole recording
    if len(segment) < 2 or segment['time'].iloc[-1] - segment['time'].iloc[0] < 1:
        return data.reset_index(drop=True)
    return segment


#Do not modify
def baseline_count_reps(data, sample_rate, exercise=None):
    # A1-style peak counting on the filtered magnitude: peaks above the mean, at least 1 s apart
    signal = data['filtered_accel_mag']
    peaks, _ = find_peaks(signal, height=signal.mean(), distance=max(int(sample_rate), 1))
    return len(peaks)


#Do not modify
def build_feature_frames(manifest, window_sec, segment_fn, feature_fn):
    # For every recording: prepare -> segment_fn -> cut -> windows of window_sec (50% overlap) -> feature_fn
    frames = []
    for _, row in manifest.iterrows():
        data, sample_rate = prepare_recording(row['path'])
        start_sec, end_sec = segment_fn(data, sample_rate)
        segment = cut_segment(data, start_sec, end_sec)

        # If the segment is shorter than one window, the whole segment is one window
        win = int(window_sec * sample_rate)
        step = max(win // 2, 1)
        for start in range(0, max(len(segment) - win, 0) + 1, step):
            window = segment.iloc[start:start + win]
            features = feature_fn(window, sample_rate)
            features['exercise'] = row['exercise']
            features['recording'] = row['path']
            frames.append(features)

    return pd.concat(frames, ignore_index=True)


#Do not modify
def select_recordings(frames, manifest):
    # Keep only the windows of the recordings in manifest
    return frames[frames['recording'].isin(manifest['path'])].reset_index(drop=True)


#Do not modify
def evaluate_rep_counter(manifest, segment_fn, count_fn, labels=None):
    # labels: optional dict {path: exercise}, e.g. your predicted exercises. By default the true exercise is used.
    rows = []
    for _, row in manifest.iterrows():
        data, sample_rate = prepare_recording(row['path'])
        start_sec, end_sec = segment_fn(data, sample_rate)
        segment = cut_segment(data, start_sec, end_sec)

        exercise = row['exercise'] if labels is None else labels[row['path']]
        rows.append({'recording': row['path'],
                     'exercise': row['exercise'],
                     'placement': row['placement'],
                     'true_reps': row['reps'],
                     'pred_reps': int(count_fn(segment, sample_rate, exercise))})

    results = pd.DataFrame(rows)
    results['error'] = results['pred_reps'] - results['true_reps']
    return results


#Do not modify
def rep_metrics(results, by=None):
    # MAE: mean absolute error in reps
    # within_2: share of recordings counted within +-2 reps of the reported count
    # bias: mean of (counted - reported); above 0 means counting too many reps
    abs_error = results['error'].abs()
    results = results.assign(abs_error=abs_error, within_2=(abs_error <= 2))

    if by is None:
        return pd.Series({'recordings': len(results),
                          'MAE': results['abs_error'].mean(),
                          'within_2': results['within_2'].mean(),
                          'bias': results['error'].mean()}).round(2)

    return results.groupby(by).agg(recordings=('error', 'size'),
                                   MAE=('abs_error', 'mean'),
                                   within_2=('within_2', 'mean'),
                                   bias=('error', 'mean')).round(2)


#Do not modify
def plot_segment(data, segments, title=''):
    # segments: dict {name: (start_sec, end_sec)}, e.g. {'baseline': baseline_segment(data, sample_rate)}
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(data['time'], data['filtered_accel_mag'], linewidth=0.8, color='gray')

    colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']
    for i, (name, (start_sec, end_sec)) in enumerate(segments.items()):
        ax.axvspan(start_sec, end_sec, color=colors[i % len(colors)], alpha=0.2, label=name)

    ax.set_xlabel('time (s)')
    ax.set_ylabel('filtered_accel_mag (m/s$^2$)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    plt.show()
