import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from itertools import combinations
import time
import torch
import torch.nn as nn

from soft_dtw_cuda import SoftDTW

import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

TracersParamsPath = 'TracersParams.csv'

#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".h5"


print("Loading data!")

dataDict = {}

loadPath = DataSavepath + f"_flat-wrt-time"+ DataSavepathSuffix

dataDict = hdf5_load(loadPath)

print(torch.__version__)

cuda = torch.device('cuda')

batch_limit = 1e6
