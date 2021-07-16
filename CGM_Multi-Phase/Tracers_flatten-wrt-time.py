import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt

import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

import multiprocessing as mp
import sys
import logging

TracersParamsPath = "TracersParams.csv"

singleValueParams = ["Lookback", "Ntracers", "Snap"]

# Number of cores to run on:
n_processes = 3

DataSavepathSuffix = f".h5"
# ==============================================================================#

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

saveParams = (
    TRACERSPARAMS["saveParams"]
    + TRACERSPARAMS["saveTracersOnly"]
    + TRACERSPARAMS["saveEssentials"]
)

for param in singleValueParams:
    saveParams.remove(param)

DataSavepathSuffix = f".h5"


def err_catcher(arg):
    raise Exception(f"Child Process died and gave error: {arg}")
    return


if __name__ == "__main__":
    print("Flattening wrt time!")
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"{rin}R{rout}")
        print("\n" + f"Sorting multi-core arguments!")

        print("Loading data!")
        args_list = []

        args_default = [
            rin,
            rout,
            TRACERSPARAMS,
            saveParams,
            DataSavepath,
            DataSavepathSuffix,
        ]
        for targetT in TRACERSPARAMS["targetTLst"]:
            dataDict = {}
            for snap in range(
                int(TRACERSPARAMS["snapMin"]),
                min(
                    int(TRACERSPARAMS["snapMax"] + 1),
                    int(TRACERSPARAMS["finalSnap"] + 1),
                ),
                1,
            ):
                loadPath = (
                    DataSavepath
                    + f"_T{targetT}_{rin}R{rout}_{int(snap)}"
                    + DataSavepathSuffix
                )
                data = hdf5_load(loadPath)
                dataDict.update(data)
            args_list.append([targetT, dataDict] + args_default)

        # Open multiprocesssing pool
        # flatten_wrt_time(4.0,rin,rout,dataDict,TRACERSPARAMS,saveParams,DataSavepath,DataSavepathSuffix)
        print("\n" + f"Opening {n_processes} core Pool!")
        pool = mp.Pool(processes=n_processes)
        print("Pool opened!")
        print("Analysis!")

        # Compute Snap analysis
        res = [
            pool.apply_async(flatten_wrt_time, args=args, error_callback=err_catcher)
            for args in args_list
        ]

        print("Analysis done!")

        pool.close()
        pool.join()
        # Close multiprocesssing pool
        print(f"Closing core Pool!")
        print(f"Final Error checks")
        success = [result.successful() for result in res]
        assert all(success) == True, "WARNING: CRITICAL: Child Process Returned Error!"
        print("Done! End of Post-Processing :)")
