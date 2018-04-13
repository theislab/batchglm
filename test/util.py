import models.stats as stats
import pandas as pd
import numpy as np


def stat_frame(obj_1, obj_2, params: list):
    df = pd.DataFrame(columns=params)
    
    nmae = []
    for i in params:
        nmae.append(stats.normalized_mae(getattr(obj_1, i), getattr(obj_2, i)))
    df.loc["normalized mean absolute error (%)", :] = np.asarray(nmae) * 100
    
    nrmsd = []
    for i in params:
        nrmsd.append(stats.normalized_rmsd(getattr(obj_1, i), getattr(obj_2, i)))
    df.loc["normalized RMSD (%)", :] = np.asarray(nrmsd) * 100
    
    df = df.round(pd.Series([1, 3], index=df.index))
    
    return df
