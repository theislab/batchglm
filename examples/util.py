import utils.stats as stats
import pandas as pd
import numpy as np


def stat_frame(obj_1, obj_2, params: list):
    df = pd.DataFrame(columns=params)

    row = []
    for i in params:
        row.append(stats.mapd(getattr(obj_1, i), getattr(obj_2, i)))
    df.loc["mean absolute percentage deviation (%)", :] = np.asarray(row) * 100

    row = []
    for i in params:
        row.append(stats.normalized_rmsd(getattr(obj_1, i), getattr(obj_2, i)))
    df.loc["normalized RMSD (%)", :] = np.asarray(row) * 100

    df = df.round(pd.Series([1, 3], index=df.index))

    return df
