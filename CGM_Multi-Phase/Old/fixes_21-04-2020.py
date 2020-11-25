for key, value in FullDict.items():
    FullDict[key]['age'] = np.array(FullDict[key]['age'][np.where(np.isnan(FullDict[key]['age'])==True)].tolist() \
    + FullDict[key]['age'][np.where(np.isnan(FullDict[key]['age'])==False)].tolist())

for key, value in FullDict.items():
    FullDict[key]['L'] *= 1e3*c.parsec*1e-5
    FullDict[key]['P_kinetic'] *= 1e10 *c.msol *1e-3
