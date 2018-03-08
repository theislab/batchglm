import pandas as pd
import numpy as np

from models.negative_binomial import RSA_negative_binomial
from impl.tf.base import BasicSession

# import matplotlib.pyplot as plt

################################
# Estimate NB distribution parameters by parameter optimization
################################

def validateData(sample_data):
    smpls = np.mean(sample_data, 0) < np.var(sample_data, 0)
    print("removing samples due to too small variance: \n%s" % np.where(smpls == False))
    
    return np.where(smpls)


if __name__ == '__main__':
    # load sample data
    sample_data = np.loadtxt("resources/sample_data.tsv", delimiter="\t")
    df = pd.read_csv("resources/sample_params.tsv", sep="\t")
    
    # smpls = np.array(range(10))
    # smpls = range(np.shape(sample_data)[1])
    smpls = validateData(sample_data)
    
    sample_data = sample_data[:, smpls]
    df = df.iloc[smpls]
    
    # previously sampled data
    
    model = RSA_negative_binomial()
    
    session = BasicSession(model, sample_data)
    session.initialize()
    session.train(1)
    session.evaluate(df)
    
    
