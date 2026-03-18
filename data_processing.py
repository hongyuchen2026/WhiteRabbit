import pandas as pd

def expand_labels(df):

    df["StartHesitation_expanded"] = df["StartHesitation"]
    freeze_index = df[df["StartHesitation"] == 1].index
    for idx in freeze_index:
        start = max(0, idx - 300)
        df.loc[start:idx, "StartHesitation_expanded"] = 1
    return df


def sliding_windows(df, window, step):

    segments = []
    labels = []
    for i in range(0, len(df) - window + 1, step):
        segment = df.iloc[i:i+window]
        label = int(segment["StartHesitation_expanded"].max())
        segments.append(segment)
        labels.append(label)

    return segments, labels