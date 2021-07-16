for key, value in FullDict.items():
    FullDict[key]["Ntracers"] = np.array([FullDict[key]["Ntracers"]])

hdf5_save(savePath, FullDict)
