import numpy as np

def weightedperc(data, weights, perc):
    
    ind_sorted = np.argsort(data)
    sorted_data = np.array(data)[ind_sorted]
    sorted_weights = np.array(weights)[ind_sorted]
    cum = np.cumsum(sorted_weights)
    whereperc, = np.where(cum/float(cum[-1]) >= perc)                                                                                           
    return sorted_data[whereperc[0]]