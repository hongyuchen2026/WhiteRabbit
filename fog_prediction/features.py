def extract_features(segment):

    feature_row = [
        segment["AccV"].mean(),
        segment["AccV"].std(),
        segment["AccV"].max(),
        segment["AccV"].min(),
        (segment["AccV"]**2).mean(),

        segment["AccML"].mean(),
        segment["AccML"].std(),
        segment["AccML"].max(),
        segment["AccML"].min(),
        (segment["AccML"]**2).mean(),

        segment["AccAP"].mean(),
        segment["AccAP"].std(),
        segment["AccAP"].max(),
        segment["AccAP"].min(),
        (segment["AccAP"]**2).mean(),

        (abs(segment["AccV"]) + abs(segment["AccML"]) + abs(segment["AccAP"])).mean()
    ]

    return feature_row