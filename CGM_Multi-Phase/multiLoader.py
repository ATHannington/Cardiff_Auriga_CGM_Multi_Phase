for dataKey in saveParams:
    print(f"{dataKey}")

    idIndent = 0
    pridIndent = 0
    tridIndent = 0

    mergedDict = {}

    for sim in simLst:

      for tindex in Tlst:
          T = Tlst[tindex]

          for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
              print(f"{rin}R{rout}")

              for (jj, snap) in enumerate(snapRange):



                TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersParamsPath)

                saveParams = TRACERSPARAMS["saveParams"]

                saveHalo = (TRACERSPARAMS["savepath"].split("/"))[-2]

                DataSavepathSuffix = f".h5"

                print("Loading data!")

                dataDict = {}

                dataDict = full_dict_hdf5_load(DataSavepath, TRACERSPARAMS, DataSavepathSuffix)

                selectKey = (
                    f"T{T}",
                    f"{rin}R{rout}",
                    f"{int(snap)}",
                )
                dataDict[selectKey]['id'] += idIndent
                dataDict[selectKey]['prid'] += pridIndent
                dataDict[selectKey]['trid'] += tridIndent

                for key,values in dataDict[selectKey].items():
                  if key in list(mergedDict.keys()):
                    mergedDict[key] = np.concatenate((mergedDict[key],values),axis=None)
                  else:
                    mergedDict.update({key : values})


      idIndent += int(np.nanmax(dataDict[selectKey]['id'])) + 1
      pridIndent += int(np.nanmax(dataDict[selectKey]['prid'])) + 1
      tridIndent += int(np.nanmax(dataDict[selectKey]['trid'])) + 1
