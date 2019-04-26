#===============================================================================#
#-------------------------------------------------------------------------------#
#		PlotProbabilities.py                                                    #
#		Plots the emission probabilities as generated by RadTrans_MainCode.F90  #
#		Details of this program below.                                          #
#-------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime
import os
from astropy import units as u

now = datetime.datetime.now()

print ()
print ("Current date and time using str method of datetime object:")
print (str(now))


##
date_last_edited= "18/04/2019"													#PLEASE KEEP THIS UP-TO-DATE!!                                                #

																				#Input directory into which to save plots here                                #
savepath = "C:/Users/C1838736/Documents/ATH_PhD/_PhD_Output/" + \
"Cylindrically_Symmetric_Filament_MCRT/Probabilities/"
																				#    Note: if func_datetime_savepath used, a subdirectory will be made here   #
																				#      using today's date at runtime.                                         #

#------------------------------------------------------------------------------#
#Files for data to be imported for plotting.                                   #
lambdaDataPath  = "WLData.csv"
#Black Body Data:
BBconstantsPath = "BBconstants.csv"
BBAnalyticPath  = "BBAnalytic.csv"
BBmcrtPath 		= "BBMCRT.csv"
#Modified Black Body data:
MBconstantsPath = "MBconstants.csv"
MBAnalyticPath  = "MBAnalytic.csv"
MBmcrtPath 		= "MBMCRT.csv"
#Differential Modified Black Body data:
DMconstantsPath = "DMconstants.csv"
DMAnalyticPath  = "DMAnalytic.csv"
DMmcrtPath 		= "DMMCRT.csv"


##

#-------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------#

# !!!
author= "Andrew T. Hannington"
email= "HanningtonAT@cardiff.ac.uk"
affiliation= "Cardiff University, Wales, UK"

adapted_from_author = "Prof. A.P. Whitworth"
adapted_from_email = "anthony.whitworth@cardiff.ac.uk"
adapted_from_affiliation = "Cardiff University, Wales, UK"

date_created= "12/04/2019"

#
# Notes: Python program for plotting emission probabilities from 
# 		 Prof. A. P. Whitworth's RadTrans MCRT code for Radially Symmetric 
#		 Filamentary Molecular Clouds.
#		 Equivalent subroutine in A.P.W's code:
#		 # SUBROUTINE RT_EmProbs_DMBB(TEkTOT,teT,WLlTOT,WLlam,WLdlam,WLchi,\
#			WLalb,PRnTOT,WTpack,WTplot,WTpBB,WTlBBlo,WTlBBup,WTpMB,WTlMBlo, \
#			WTlMBup,teLMmb,WTpDM,WTlDMlo,WTlDMup,teLMTdm)
#
#-------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
#		Below are functions etc. used by this program                           #
#                                                                               #
#                                                                               #


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
def func_datetime_savepath (input_savepath_string):
	"""
	Description: Function for generating a savepath string and creating 
				subsequent directory.
				NOTE: this function will NOT create all intermediate level 
				directories in path name. To do this, please see Python 
				documentation on os.makedirs()
	Inputs:		Var: input_savepath_string	Type:string 	Dtype: char
	Outputs:	Var: savepath				Type: string	Dtype: char
				------
	Notes:		Created 09/04/2019 by ATH. Working as of 09/04/2019
	"""
	
	save_date = str(now.strftime("%Y-%m-%d-%H-%M"))
	savepath = input_savepath_string + "/" + save_date +"/"
	print()
	print("Savepath generated! Datetime used!")
	os.mkdir(savepath)
	print("Directory created at savepath!")
	print("Your savepath directory path is:")
	print(savepath)
	print()
	return savepath

#-------------------------------------------------------------------------------#
#		Below is the beginning of the program which writes to screen the        #
#		information about authors etc. above.                                   #
#                                                                               #
#                                                                               #
print("*****")
print()
print("**Plot Emission Probabilities Program**")
print()
print(f"Author: {author}")
print(f"Email: {email}")
print(f"Affiliation: {affiliation}")
print()
print(f"Adapted from work by: {adapted_from_author}")
print(f"Email: {adapted_from_email}")
print(f"Affiliation: {adapted_from_affiliation}")
print()
print(f"Program first created: {date_created}")
print(f"Program last edited: {date_last_edited}")
print()
print("*****")
print()

del author, email, affiliation, adapted_from_affiliation, adapted_from_author \
,adapted_from_email, date_created, date_last_edited								#Clear preamble variables from memory                                         #

#-------------------------------------------------------------------------------#
#		Begin Program by loading in data from RadTrans_MainCode.F90.            #
#                                                                               #
#                                                                               #

datetimesavepath = func_datetime_savepath(savepath)								#Generate savepath for plots												  #

print()
print("Loading in data!")


																				#Read in Wavelength Data into single columned Data Frame.					  #
																				# We have omitted any infered header(column title) and skipped any white space#
lambdaData 	 = pd.read_csv(lambdaDataPath, header=None, \
skipinitialspace =True)

###Load into Data frame, omit header for data, skip spaces, 
###    and tranpose for ease of plotting
BBconstants  = pd.read_csv(BBconstantsPath,delimiter=" ")						#We read the constants data into a Pandas Data Frame                          #
																				#We read in the analytic and MCRT data,currently in format of one row per temp#
																				#  we omit headers again [this would just be indices], skip whitespace, and   #
																				#    set any Fortran90 +/-"Infinity" values to NaN.							  #
																				#      We transpose the data so each data set is in one temperature index col #
																				#        i.e. one column per temperature, labelled by temp number [0,1,2,3...]#
BBAnalytic	 = pd.read_csv(BBAnalyticPath,delimiter=" ",header=None, \
skipinitialspace =True, na_values=["-Infinity", "Infinity"]).T
BBMCRT		 = pd.read_csv(BBmcrtPath,delimiter=" ",header=None, \
skipinitialspace =True, na_values=["-Infinity", "Infinity"]).T

MBconstants  = pd.read_csv(MBconstantsPath,delimiter=" ")						
MBAnalytic	 = pd.read_csv(MBAnalyticPath,delimiter=" ",header=None, \
skipinitialspace =True, na_values=["-Infinity", "Infinity"]).T
MBMCRT		 = pd.read_csv(MBmcrtPath,delimiter=" ",header=None, \
skipinitialspace =True, na_values=["-Infinity", "Infinity"]).T

DMconstants  = pd.read_csv(DMconstantsPath,delimiter=" ")						
DMAnalytic	 = pd.read_csv(DMAnalyticPath,delimiter=" ",header=None, \
skipinitialspace =True, na_values=["-Infinity", "Infinity"]).T
DMMCRT		 = pd.read_csv(DMmcrtPath,delimiter=" ",header=None, \
skipinitialspace =True, na_values=["-Infinity", "Infinity"]).T


print("Data loading complete!")

print()
print("Manipulating data!")
###
###Cut off end NaN from Fortran Carriage Return Space							
###
																				#Here we have used online resources to determine the method by which to omit  #
																				#  the last row of the data. In this instance, the last row is a NaN brought  #
																				#    about by the space at each carriage return from Fortran90. Without this  #
																				#     fix then there is +1 "entry", creating a mismatch in len with wavelength#
BBAnalytic.drop(BBAnalytic.tail(1).index,inplace=True)
BBMCRT.drop(BBMCRT.tail(1).index,inplace=True)
MBAnalytic.drop(MBAnalytic.tail(1).index,inplace=True)
MBMCRT.drop(MBMCRT.tail(1).index,inplace=True)
DMAnalytic.drop(DMAnalytic.tail(1).index,inplace=True)
DMMCRT.drop(DMMCRT.tail(1).index,inplace=True)


																				#---CURRENTLY OMITTED CODE---												  #
																				#This section updates each columns of each data data-frame with a float64     #
																				#  version of the data. It "coerces" any non-numeric values to NaNs.		  #
# ###Replace all values with float64 and "coerce" unrecognised into NaNs
# for cols in BBAnalytic.columns.values:
	# BBAnalytic[cols] = pd.to_numeric(BBAnalytic[cols], errors='coerce')
	# BBMCRT[cols] = pd.to_numeric(BBMCRT[cols], errors='coerce')
	# MBAnalytic[cols] = pd.to_numeric(MBAnalytic[cols], errors='coerce')
	# MBMCRT[cols] = pd.to_numeric(MBMCRT[cols], errors='coerce')
	# DMAnalytic[cols] = pd.to_numeric(DMAnalytic[cols], errors='coerce')
	# DMMCRT[cols] = pd.to_numeric(DMMCRT[cols], errors='coerce')

#-------------------------------------------------------------------------------#
																				# Debugging print statements...												  #
# print()
# print("***")
# print("Data Details (for debugging):")
# print()
# print()
# print("Wavelength:")
# print(lambdaData,np.shape(lambdaData))
# print()
# print("BB:")
# print("Constants:",BBconstants,np.shape(BBconstants),BBconstants.columns.values)
# print("Analytic:",BBAnalytic,np.shape(BBAnalytic),BBAnalytic.columns.values)
# print("MCRT:",BBMCRT,np.shape(BBMCRT),BBMCRT.columns.values)
# print()
# print("MB:")
# print("Constants:",MBconstants,np.shape(MBconstants),MBconstants.columns.values)
# print("Constants:",MBconstants,np.shape(MBconstants),MBconstants.columns.values)
# print("Analytic:",MBAnalytic,np.shape(MBAnalytic),MBAnalytic.columns.values)
# print("MCRT:",MBMCRT,np.shape(MBMCRT),MBMCRT.columns.values)
# print()
# print("DM(BB):")
# print("Constants:",DMconstants,np.shape(DMconstants),DMconstants.columns.values)
# print("Analytic:",DMAnalytic,np.shape(DMAnalytic),DMAnalytic.columns.values)
# print("MCRT:",DMMCRT,np.shape(DMMCRT),DMMCRT.columns.values)



#-------------------------------------------------------------------------------
																				#Here we find the "limiting ordinates" (max and min for each axis).These are  #
																				#  part of the constants file, but the max and min must reflect the max and   #
																				#    min of all temperatures being shown so as to not omit any data.          #
																				#      It should be noted that the max and min cut off the lower y-axis, where#
																				#       results start to become spurious, and there is no available MCRT data.#
BBxMin = np.min(BBconstants['xmin'])
BBxMax = np.max(BBconstants['xmax'])
BByMin = np.min(BBconstants['ymin'])
BByMax = np.max(BBconstants['ymax'])

MBxMin = np.min(MBconstants['xmin'])
MBxMax = np.max(MBconstants['xmax'])
MByMin = np.min(MBconstants['ymin'])
MByMax = np.max(MBconstants['ymax'])

DMxMin = np.min(DMconstants['xmin'])
DMxMax = np.max(DMconstants['xmax'])
DMyMin = np.min(DMconstants['ymin'])
DMyMax = np.max(DMconstants['ymax'])

print()
print("Beginning Plotting!")

																				#A fairly standard plotting section. Each plot is overlayed with the MCRT and #
																				#  analytic data from each temperature. Each curve is labelled appropriately  #
																				#    with temperature and data type (analytic vs MCRT) using f-strings.       #
																				#     Xlims and Ylims set from above and in Fortran90 code. Plots are then    #
																				#      titled, labelled, legend added, and saved in date-time directory.      #


fig = plt.figure()
for i in range(0, len(BBconstants['temp'])):
	plt.plot(lambdaData,BBAnalytic[i], label=\
	f"BB Analytic T={BBconstants['temp'][i]}K")
	plt.plot(lambdaData,BBMCRT[i],label=\
	f"BB MCRT T={BBconstants['temp'][i]}K")
plt.xlabel('Log10(Wavelength) [Log10(microns)]')
plt.ylabel('Log10(BB Probability) [dimensionless]')
plt.title('Log10(Black Body Probabilities) against Log10(Wavelength)'+'\n'\
+ 'for given temperatures')
plt.xlim(BBxMin, BBxMax)
plt.ylim(BByMin, BByMax)

fig.legend(bbox_to_anchor=(.85,.85), loc="upper right", borderaxespad=0.)
fig.savefig(datetimesavepath+\
'Log10(BB)_mcrt-vs-analytic_vs_Log10(wavelength)' + '.jpeg')

fig = plt.figure()
for i in range(0, len(MBconstants['temp'])):
	plt.plot(lambdaData,MBAnalytic[i],label=\
	f"MB Analytic T={BBconstants['temp'][i]}K")
	plt.plot(lambdaData,MBMCRT[i],label=\
	f"MB MCRT T={MBconstants['temp'][i]}K")
plt.xlabel('Log10(Wavelength) [Log10(microns)]')
plt.ylabel('Log10(MB Probability) [dimensionless]')
plt.title('Log10(Modified Black Body Probabilities) against Log10(Wavelength)'+\
'\n' + 'for given temperatures')
plt.xlim(MBxMin, MBxMax)
plt.ylim(MByMin, MByMax)

fig.legend(bbox_to_anchor=(.85,.85), loc="upper right", borderaxespad=0.)
fig.savefig(datetimesavepath+\
'Log10(MB)_mcrt-vs-analytic_vs_Log10(wavelength)'+'.jpeg')

fig = plt.figure()
for i in range(0, len(DMconstants['temp'])):
	plt.plot(lambdaData,DMAnalytic[i],label=\
	f"DM Analytic T={DMconstants['temp'][i]}K")
	plt.plot(lambdaData,DMMCRT[i], label=\
	f"DM MCRT T={DMconstants['temp'][i]}K")
plt.xlabel('Log10(Wavelength) [Log10(microns)]')
plt.ylabel('Log10(DMBB Probability) [dimensionless]')
plt.title('Log10(Differential Modified Black Body Probabilities)' + '\n'\
+'against Log10(Wavelength)'+'\n'+ 'for given temperatures')
plt.xlim(DMxMin, DMxMax)
plt.ylim(DMyMin, DMyMax)

fig.legend(bbox_to_anchor=(.85,.85), loc="upper right", borderaxespad=0.)
fig.savefig(datetimesavepath+\
'Log10(DMBB)_mcrt-vs-analytic_vs_Log10(wavelength)'+'.jpeg')

finxMin = min([DMxMin, MBxMin])
finxMax = max([DMxMax, MBxMax])
finyMin = min([DMyMin, MByMin])
finyMax = max([DMyMax, MByMax])

fig = plt.figure()
for i in range(0, len(MBconstants['temp'])):
	plt.plot(lambdaData,DMAnalytic[i],label=\
	f"DM Analytic T={DMconstants['temp'][i]}K")
	plt.plot(lambdaData,MBAnalytic[i], label=\
	f"MB Analytic T={MBconstants['temp'][i]}K")
plt.xlabel('Log10(Wavelength) [Log10(microns)]')
plt.ylabel('Log10(DMBB Probability) [dimensionless]')
plt.title('Log10(Differential Modified Black Body Probabilities)' + '\n'\
+'& Log10(Modified Black Body Probabilities)'+'\n' \
+'against Log10(Wavelength)for given temperatures')
plt.xlim(DMxMin, DMxMax)
plt.ylim(DMyMin, DMyMax)

fig.legend(bbox_to_anchor=(.85,.85), loc="upper right", borderaxespad=0.)
fig.savefig(datetimesavepath \
+'Log10(MB)-analytic_Log10(DMBB)-analytic_vs_Log10(wavelength)' +'.jpeg')

plt.show()

print()
print("END")																	#---PROGRAM END!---													 		  #
