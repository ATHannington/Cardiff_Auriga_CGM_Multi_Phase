#===============================================================================#
#-------------------------------------------------------------------------------#
#		PlotDustProperties.py                                                   #
#		Plots the properties of dust as generated by RadTrans_MainCode.F90      #
#		Details of this program below.                                          #
#-------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import pandas as pd
import math
import datetime
import os
from astropy import units as u
from astropy.modeling.blackbody import blackbody_lambda
#from astropy.modeling.blackbody import blackbody_lambda


now = datetime.datetime.now()

print ()
print ("Current date and time using str method of datetime object:")
print (str(now))

# print(np.__version__)

# !!!
author= "Andrew T. Hannington"
email= "HanningtonAT@cardiff.ac.uk"
affiliation= "Cardiff University, Wales, UK"

adapted_from_author = "Prof. A.P. Whitworth"
adapted_from_email = "anthony.whitworth@cardiff.ac.uk"
adapted_from_affiliation = "Cardiff University, Wales, UK"

date_created= "05/04/2019"

#
# Notes: Python program for plotting dust properties from
# 		 Prof. A. P. Whitworth's RadTrans MCRT code for Radially Symmetric
#		 Filamentary Molecular Clouds.
#		 Equivalent subroutine in A.P.W's code:
#		 # SUBROUTINE RT_PlotDustProperties(WLlTOT,WLlam,WLchi,WLalb)
#
##
##
date_last_edited= "12/04/2019"													#PLEASE KEEP THIS UP-TO-DATE!!                                                #

																				#Input directory into which to save plots here                                #
savepath = "./media/sf_OneDrive_-_Cardiff_University/Documents/ATH_PhD/"+ \
"_PhD_Output/Cylindrically_Symmetric_Filament_MCRT/Dust"
																				#    Note: if func_datetime_savepath used, a subdirectory will be made here   #
																				#      using today's date at runtime.                                         #

importstring = "DustProperties.csv"												#File for data to be imported for plotting.                                   #

temperatures = np.array([3.16,10.0,31.6,100.,316.])*u.K							#Select temperatures to plot black body curves and modified black body curves #
##
##

#!!!

#-------------------------------------------------------------------------------#
#		Below are constants used by this program                                #
#			Here we have used the AstroPy module to allocate units and convert  #
#				to CGS units.                                                   #
#                                                                               #
h = 6.62607004e-34*u.m**2 * u.kg / u.s											#Planck constant [m^2 kg / s]
h = h.cgs																		#Planck constant to CGS units [cm^2 g / s]
#print("TEST: Planck Constant in CGS =", h)

c  = 299792458 * u.m / u.s														#Speed of Light [m / s]
c = c.cgs																		#Speed of Light to CGS units [cm /s]
#print("TEST: C in CGS =", c)

kb = 1.3806852e-23 * u.m**2 * u.kg / (u.s**2 * u.K)								#Boltzmann constant [m^2 kg / s^2 /K ]
kb = kb.cgs
#print("TEST: kb in CGS =", kb)

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

	save_date = str(now.strftime("%Y-%m-%d"))#-%H-%M"))
	savepath = input_savepath_string + "/" + save_date +"/"
	print()
	print("Savepath generated! Datetime used!")
	os.mkdir(savepath)
	print("Directory created at savepath!")
	print("Your savepath directory path is:")
	print(savepath)
	print()
	return savepath
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
def func_BB_lam(lambda_arr,hc_kbt_arr):
	"""
	Description: Function for generating a 2D NumPy array of Black Body Spectral
				Density, as a function of wavelength, and un-normalised.
				hc_kbt_arr input should be a 2D [n,m] array h*c/(kb*T) and
				shaped:
				[ T_1 T_1 ... ->]
				[ T_2 T_2 ...	]
				[  v   v  ...	]
				[ T_n T_n ...T_n]
				lambda_arr input should be a 2D [n,m]array of wavelengths in
				appropriate units and shaped:
				[ l_1 l_2 ->	]
				[ l_1 l_2 ...	]
				[  v   v  ...	]
				[ l_1 l_2 ...l_m]
	Inputs:		Var: lambda_arr 	Type:numpy array 2D [n,m] 	Dtype: float
				Var: hc_kbt_arr 	Type:numpy array 2D [n,m] 	Dtype: float
	Outputs:	Var: bb_data 		Type:numpy array 2D [n,m] 	Dtype: float

				------
	Notes:		Created 11/04/2019 by ATH. Working as of 11/04/2019
	"""

	bb_data = 1./((lambda_arr**5)*(np.exp(hc_kbt_arr*(1./lambda_arr.value))-1.))
	return bb_data


#-------------------------------------------------------------------------------#
#		Below is the beginning of the program which writes to screen the        #
#		information about authors etc. above.                                   #
#                                                                               #
#                                                                               #
print("*****")
print()
print("**Plot Dust Properties Program**")
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

read_data = pd.read_csv(importstring,delimiter=" ", \
skipinitialspace =True, na_values=["-Infinity", "Infinity"])					#We read the data into a Pandas Data Frame                                    #

print()
print("File read from:", importstring)
del importstring																#Clear importstring from memory                                               #
print("Data Frame shape:")
print(read_data.shape)															#Confirm data is of correct shape                                             #
print(read_data.columns.values)
#print(read_data)
print()


#-------------------------------------------------------------------------------#
# 						Data Manipulation Section							    #
#																			    #
#-------------------------------------------------------------------------------#
print()
print("Manipulating in data! [part 1/3]")

																				#Create lists of Log10(Lambda),                                               #
#print("Test! lam(0)=", read_data['lam'][0])

																				#A rather involved process... We convert the values to Log10 values and then  #
x_lamLog10 = np.log10(read_data['lam'])											#  check whether the return are inf or NaN. Replace with NaN if so as can be  #
																				#    ignored through numpy's NaNmax and NaNmin routines, and otherwise aren't
																				#      plotted.
x_lamLog10 = np.where(((np.isnan(x_lamLog10)==False)&\
(np.isinf(x_lamLog10)==False)),x_lamLog10,float('NaN'))

																				#    and of Log10(Chi).                                                       #
#print("Test! chi(0)=", read_data['chi'][0])
y_chiLog10 = np.log10(read_data['chi'])
y_chiLog10 = np.where(((np.isnan(y_chiLog10)==False)&\
(np.isinf(y_chiLog10)==False)),y_chiLog10,float('NaN'))

xmin = np.nanmin(x_lamLog10)													#  then find max and min values for each, to use for axes in plots, ignoring  #
xmax = np.nanmax(x_lamLog10)													#    any NaNs.																  #
ymin = np.nanmin(y_chiLog10)
ymax = np.nanmax(y_chiLog10)

																				#Create a list of 10 times the albedo. This is used to bring the Albedo up to #
																				#    a comparable scale with Log10(chi) and Log10(lambda).                    #

z_10alb = [(10*read_data['alb'][i]) for i in \
range(0,len(read_data['alb']),1)]

zmin = min(z_10alb)
zmax = max(z_10alb)

deltafrac = 0.1																	#Set some fraction of delta x or y (max-min) to pad axes by.                  #

deltax = deltafrac*(xmax-xmin)													#Compute updated xmin, xmax, ymin, ymax, updated by axis padding.             #
xmax = xmax + deltax
xmin = xmin - deltax
deltay = deltafrac*(ymax-ymin)
ymax = ymax + deltay
ymin = ymin - deltay
deltaz = deltafrac*(zmax-zmin)
zmax = zmax + deltaz
zmin = zmin - deltaz


del deltafrac,deltax,deltay,deltaz												#Clear deltafrac, deltax, deltay from memory                                  #

#-------------------------------------------------------------------------------#
#           Begin Plotting														#
#                                                                               #
#		Much of the following has been adapted from the Matplotlib documentation#
#                                                                               #

print()
print("Creating first plots!")

fig, ax1 = plt.subplots()														#Start new fig and axis with plt.subplots()                                   #

color = 'red'																	#Set first colour                                                             #
plot2 = ax1.plot(x_lamLog10, y_chiLog10, color=color, label = r"$\chi$: Absorption Coeff. per unit mass")			#    Plot Log10(Chi) against Log10(Wavelength), in above colour,
																				#        with legend label "chi".											  #
ax1.set_xlim(xmin,xmax)															#        Set axes to have max and min values as calculated                    #
ax1.set_ylim(min(ymin,zmin),max(ymax,zmax))
ax1.set_xlabel(r"$Log_{10}(\lambda)$ [$\mu$m]")									#            Set x axis label, with units                              		  #
ax1.set_ylabel(r"$Log_{10}(\chi)$ [$cm^{2}$ $g^{-1}$]",color=color)				#                Set y axis label, with units, in color defined above.        #

ax2 = ax1.twinx()																#Set second axis plot to have a twin of the first's x-axis.                   #

color='blue'
																				#Plot 10*Albedo against Log10(Wavelength), dashed, in colour 2, legend label  #
																				#  "Albedo".
plot1 = ax2.plot(x_lamLog10, z_10alb,color=color, linestyle="-" ,\
label = r"$10*a$: albedo")
ax2.set_xlim(xmin,xmax)															#    Set axes limits to min  and max of new variables                         #
ax2.set_ylim(min(ymin,zmin),max(ymax,zmax))
ax1.set_xlabel(r"$Log_{10}(\lambda)$ [$\mu$m]")									#            Set x axis label, with units                              		  #
ax2.set_ylabel(r"$10*a$ [/]", color=color)

plt.title(r"Comparative Plot of $Log_{10}$(Absorption Coeff. per unit mass) and 10*Albedo" + "\n" \
+ r"versus $Log_{10}$(Wavelength)")
fig.legend(bbox_to_anchor=(.85,.85), loc="upper right", borderaxespad=0.)		#Attach Legend to image, in adjusted upper right [NOTE: Standard Upper Right  #
																				#  does not work here. Ejects image out of frame, overlapping with 2nd y-axis #
																				#    tickers]. We have added "borderaxespad=0." to remind self option exists. #
fig.tight_layout()																#According to Maplotlib documentation this line is needed to ensure the second#
																				#  axis does not become clipped.                                              #

ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.grid(which="both")
plt.show()																		#Show figure!                                                                 #


# datetimesavepath = func_datetime_savepath(savepath)								#Create dated subdirectory and savepath directory path.                       #
fig.savefig("Log10-chi_10Albedo_versus" + \
"_Log10-Wavelength.png")




#-------------------------------------------------------------------------------#
#		Second data manipulation.                                               #
#           Plotting for Volume Emissivity and Planck Functions!                #
#                                                                               #
#                                                                               #

print()
print("Manipulating data [part 2/3]!")

# ntemps = len(temperatures)														#Grab number of temperatures used                                             #
# nlambda = len(read_data['lam'])													#Grab number of wavelength data points                                        #
# hc_kb = h*c / kb																#Calculate constant                                                           #

lam = np.array(read_data['lam'])*u.micron
lam = (lam).to(u.angstrom)														#convert to angstrom for astropy blackbody_lambda (required)

																				#remove dimensions of ergs, per Second per steradian
																				#will then have cm^-4 angs^-1
																				# convert entire value to  mcirons^-5
																				# have one row per temperature
planck_arr = np.array([((1./(h*(c**2)))*(4.*math.pi*(u.steradian))*\
(blackbody_lambda(lam,temperatures[t]))).to(u.micron**(-5)) \
for t in range(len(temperatures))])

# print((((1./(h*(c**2)))*(4.*math.pi*(u.steradian))*(blackbody_lambda(lam,temperatures[0]))[0])).to(u.micron**(-5)))
lam = (lam).to(u.micron)														#convert back to microns for plots
lam_tmp = lam.value
x_lamLog10 = np.log10(lam_tmp)
																				# Same as above (only way it would work) but take .value of entire entry
planck_arr_dimensionless = np.array([(((1./(h*(c**2)))*(4.*math.pi*(u.steradian))*\
(blackbody_lambda(lam,temperatures[t]))).to(u.micron**(-5))).value \
for t in range(len(temperatures))])
# planck_arr_dimensionless = np.array([ for t in range(len(temperatures))])
y_PlanckLog10 = np.log10(planck_arr_dimensionless)
#
#
# lambda_arr = np.full((ntemps,nlambda),read_data['lam'])							#Temporary array of lambda data. Needed for black body function.              #
# lambda_arr = lambda_arr*u.micron												#    Shape [temp,lambda]. Give units of microns.                              #
# lambda_arr = lambda_arr.cgs														#        Convert microns to CGS.											  #
# #-------------------------------------------------------------------------------#
#
# lambda_T = hc_kb/temperatures													#Calculate (h*c)/(kb*T). This will be shape [temp]. Needed for exponent#
# 																				#    in Black Body function.											  #
#
# lambdaMin = (lambda_T.value)*0.03														#Lambda min and max set as in A. P. Whitworth's original code.				  #
# lambdaMax = (lambda_T.value)*10.0
#
# hc_kbt_arr = (np.full((nlambda,ntemps),lambda_T)).T								#Make exponent data into Matrix form. Final shape [temp,lambda]. Needs to be  #
# 																				#  input as [lambda,temp] and transposed to get obtain correct shape without  #
# 																				#    chopping data or throwing errors.
# planck_arr = func_BB_lam(lambda_arr,hc_kbt_arr)									#Put exponent and lambda data through un-normalised Black Body function.      #
# planck_units = planck_arr.unit													#    Grab units of Planck data for re-assignment to data later.				  #
# #print(planck_units)
#
# lam = lambda_arr[0]																#Lambda array no longer needed, so one copy of the lambda data taken as 1D    #
# 																				#  array, and 2D array deleted.												  #
# del lambda_arr																	#    ^^^.																	  #
#
# planck_masked = np.zeros((ntemps,nlambda))										#Setup blank array for masked planck data. Shape [temps,lambda].			  #
# for index in range(0,ntemps,1):													#[FOR] indices up to number of temperatures [THEN]							  #
# 																				#  assign to temp column of planck_masked array, the set of planck data that  #
# 																				#    is bounded by Lambda>= Lambda_min & Lambda<=Lambda_max [ELSE]
# 	planck_masked[index] = (np.where((lam.value >= lambdaMin[index].value)\
# 	&(lam.value<=lambdaMax[index].value),planck_arr[index],None))				#      fill else entries with NaN.											  #
# 																				#NOTE: "&" has to be used (not "and") to get "index wise" entries that works  #
# 																				#  without throwing error. If not, an error about truthyness of multiple entry#
# 																				#    arrays will be thrown. 												  #
# planck_masked = planck_masked*planck_units										#Give the masked array correct units for debugging or future use.			  #

# print("lambdaMin",lambdaMin,np.shape(lambdaMin))
# print("planck_masked",planck_masked,planck_masked[0],\
# np.shape(planck_masked),planck_masked.unit)
#
# y_PlanckLog10=np.log10(planck_masked.value)										#np.Log10 to take the Log10 of every entry in the array. ".value" must be     #
# 																				#    used as the data technically still has units (contrary to definition of  #
# 																				#      a logarithm.															  #
# 																				#        We have taken the log to ???										  #
#
# yMax = (np.full((nlambda,ntemps),np.nanmax(y_PlanckLog10,axis=1))).T			#Find the maximum NON-NaN value of each temperature column of the Log10 planck#
# 																				#  data. This is then copied into an array, with each temp column holding     #
# 																				#    nlambda copies of this maximum. This is inefficient, but makes it easy to#
																				#      subtract this maximum from the entire set of PlanckLog10 data.		  #
# print("yMax",yMax,yMax[0],np.shape(yMax))
#
# y_PlanckLog10 = y_PlanckLog10 - yMax											#This subtraction effectively "normalises" the PlanckLog10 data.			  #
#
# del hc_kb, lambda_T, yMax

#-------------------------------------------------------------------------------#
																				#This section near entirely follows the section above's logic. Please see     #
																				#  above																	  #

print()
print("Manipulating data [part 3/3]!")

albedo = np.array(read_data['alb'])*u.dimensionless_unscaled
chi = np.array(read_data['chi'])*((u.cm**2)/(u.g))
modPlanck_arr = chi*(1.0 - albedo)*planck_arr
modPlanck_arr_dimensionless = chi.value*(1.0 - albedo.value)*planck_arr_dimensionless
z_modPlanckLog10 = np.log10(modPlanck_arr_dimensionless)
#
#
# alb_squareArr = np.full((ntemps,nlambda),read_data['alb'])\
# *u.dimensionless_unscaled														#[temps,lambda] shaped array of dimensionless albedo data.					  #
#
# chi_squareArr = (np.full((ntemps,nlambda),read_data['chi'])*((u.cm**2)/(u.g)))\
# .cgs																			#[temps,lambda] shaped array of chi (extinction opacity) data with correct	  #
# 																				#  units																	  #
#
# z_vol_emiss = planck_arr*chi_squareArr*(1.-alb_squareArr)						#Combine planck data with chi and albedo to obtain volume emissivity		  #
# 																				#  units [1/(cm^3 g)]														  #
# vol_emiss_units = z_vol_emiss.unit
#
# z_vol_emiss_masked = np.zeros((ntemps,nlambda))
# for index in range(0,ntemps,1):
# 	z_vol_emiss_masked[index] = (np.where((lam.value >= lambdaMin[index].value)\
# 	&(lam.value<=lambdaMax[index].value),z_vol_emiss[index],None))
#
# z_vol_emiss_masked = z_vol_emiss_masked*vol_emiss_units
#
# #print("z_vol_emiss_masked",z_vol_emiss_masked,z_vol_emiss_masked[0],\
# #np.shape(z_vol_emiss_masked),z_vol_emiss_masked.unit)
#
# z_VELog10=np.log10(z_vol_emiss_masked.value)
#
# zMax = (np.full((nlambda,ntemps),np.nanmax(z_VELog10,axis=1))).T
#
# #print("zMax",zMax,zMax[0],np.shape(zMax))
#
# z_VELog10 = z_VELog10 - zMax
#
# del alb_squareArr, chi_squareArr, z_vol_emiss, z_vol_emiss_masked, zmax			#Delete obselete data.														  #

#-------------------------------------------------------------------------------#

print()
print("Creating final (second) set of plots!")

fig, ax = plt.subplots()																#Creating figure helps prevent blanck outputs for fig.savefig().			  #
for temp in range(0,ntemps,1):													#[FOR] indices up to number of temperatures [THEN]							  #
																				#  plot Log10Planck data for a given temperature against log10(wavelength)	  #
	ax.plot(x_lamLog10,y_PlanckLog10[temp],label = \
	r"$Log_{10}(B_{\lambda})$"+f" at {temperatures[temp]:.2f}")



ax.set_xlabel(r"$Log_{10}(\lambda)$ [$\mu m$]")								#Give plot relevant axes labels and a title.								  #
ax.set_ylabel(r"$Log_{10}(B_{\lambda})$ [$\mu m^{-5}$]")
ax.set_title(r"$Log_{10}$(Planck Function) versus $Log_{10}$(Wavelength)")
ax.set_ylim(bottom=math.log10(1e-20),top=math.log10(1e5))
ax.grid(which="both")
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.legend(loc='upper right')																	#Display plot legend. 														  #
plt.show()																		#Show plot on screen.														  #

fig.savefig("Log10-planck_versus_Log10-Wavelength.png")														#Save plot in savepath directory.											  #

fig, ax = plt.subplots()
																				#As above but now for Log10 of Volume Emissivity.							  #
for temp in range(0,ntemps,1):
	ax.plot(x_lamLog10,z_modPlanckLog10[temp],label = \
	r"$Log_{10}(B^{Mod.}_{\lambda})$"+f" at {temperatures[temp]:.2f}")
plt.xlabel(r"$Log_{10}(\lambda)$ [$\mu m$]")								#Give plot relevant axes labels and a title.								  #
plt.ylabel(r"$Log_{10}(B^{Mod.}_{\lambda})$ [$\mu m^{-5}$]")
ax.set_title(r"$Log_{10}$(Modified Planck Function) versus $Log_{10}$(Wavelength)")
ax.set_ylim(bottom=math.log10(1e-20),top=math.log10(1e5))
ax.grid(which="both")
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.legend(loc='upper right')
plt.show()

fig.savefig("Log10-Volume-Emissivity_versus_Log10-Wavelength.png")


print("END")																	#----  END OF PROGRAM !! ----                                                 #
#===============================================================================#





#===============================================================================#
#				Original FORTRAN90 Code by Prof. A. P. Whitworth				#
#						Extracted 05/04/2019									#
#																				#
#																				#
#===============================================================================#





# ! !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ! SUBROUTINE RT_PlotDustProperties(WLlTOT,WLlam,WLchi,WLalb)
# ! !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ! ! This subroutine plots the optical properties of the dust grains, and
# ! ! modified Planck spectra at a selection of temperatures. It is given:
# ! !   the number of wavelengths            (WLlTOT);
# ! !   the discrete wavelengths             (WLlam(1:WLlTOT));
# ! !   the discrete extinction opacities    (WLchi(1:WLlTOT));
# ! !   and the discrete albedos             (WLalb(1:WLlTOT)).

# ! IMPLICIT NONE                                            ! [] DECLARATIONS
# ! INTEGER,     INTENT(IN)                     :: WLlTOT    ! the number of discrete wavelengths
# ! REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT) :: WLlam     ! the discrete  wavelengths
# ! REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT) :: WLchi     ! the discrete extinction opacities
# ! REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT) :: WLalb     ! the discrete albedos
# ! INTEGER                                     :: TEk       ! a dummy temperature ID
# ! REAL(KIND=8),DIMENSION(1:5)                 :: teT       ! the prescribed temperatures
# ! INTEGER                                     :: WLl       ! a dummy wavelength ID
# ! REAL(KIND=8)                                :: WLlamMAX  ! the maximum wavelength
# ! REAL(KIND=8)                                :: WLlamMIN  ! the minimum wavelength
# ! REAL(KIND=8)                                :: ZZdumR    ! a dummy real
                                                         # ! ! FOR PGPLOT
# ! REAL(KIND=4),DIMENSION(1:WLlTOT)            :: PGx       ! array for abscissa (log10[lam])
# ! REAL(KIND=4)                                :: PGxMAX    ! upper limit on abscissa
# ! REAL(KIND=4)                                :: PGxMIN    ! lower limit on abscissa
# ! REAL(KIND=4),DIMENSION(1:WLlTOT)            :: PGy       ! array for ordinate (log10[chi,PlanckFn])
# ! REAL(KIND=4)                                :: PGyMAX    ! upper limit on ordinate
# ! REAL(KIND=4)                                :: PGyMIN    ! lower limit on ordinate
# ! REAL(KIND=4),DIMENSION(1:WLlTOT)            :: PGz       ! array for ordinate (log10[alb,VolEm])
# ! REAL(KIND=4)                                :: PGzMAX    ! upper limit on ordinate
# ! REAL(KIND=4)                                :: PGzMIN    ! lower limit on ordinate

                                                         # ! ! [] DUST PROPERTIES
# ! PGxMIN=+0.1E+11                                          ! set PGx_MIN to improbably high value
# ! PGxMAX=-0.1E+11                                          ! set PGx_MAX to improbably low value
# ! PGyMIN=+0.1E+11                                          ! set PGy_MIN to improbably high value
# ! PGyMAX=-0.1E+11                                          ! set PGy_MAX to improbably low value
# ! DO WLl=1,WLlTOT                                          ! start loop over wavelengths
  # ! PGx(WLl)=LOG10(WLlam(WLl))                             !   compute LOG10[lam]
  # ! IF (PGx(WLl)<PGxMIN) PGxMIN=PGx(WLl)                   !   reduce PGx_MIN, as appropriate
  # ! IF (PGx(WLl)>PGxMAX) PGxMAX=PGx(WLl)                   !   increase PGx_MAX, as appropriate
  # ! PGy(WLl)=LOG10(WLchi(WLl))                             !   compute LOG10[chi]
  # ! IF (PGy(WLl)<PGyMIN) PGyMIN=PGy(WLl)                   !   reduce PGy_MIN, as appropriate
  # ! IF (PGy(WLl)>PGyMAX) PGyMAX=PGy(WLl)                   !   increase PGy_MAX, as appropriate
  # ! PGz(WLl)=10*WLalb(WLl)                                 !   compute 10 x alb
# ! ENDDO                                                    ! end loop over wavelengths
# ! ZZdumR=0.1*(PGxMAX-PGxMIN)                               ! compute margin for abscissa
# ! PGxMIN=PGxMIN-ZZdumR                                     ! compute minimum abscissa
# ! PGxMAX=PGxMAX+ZZdumR                                     ! compute maximum abscissa
# ! ZZdumR=0.1*(PGyMAX-PGyMIN)                               ! compute margin for ordinate
# ! PGyMIN=PGyMIN-ZZdumR                                     ! compute minimum ordinate
# ! PGyMAX=PGyMAX+ZZdumR                                     ! compute maximum ordinate
# ! WRITE (*,*) ' '                                          ! print blank line
# ! CALL PGBEG(0,'/XWINDOW',1,1)                             ! open PGPLOT to display on screen
# ! !CALL PGBEG(0,'/PS',1,2)                                  ! open PGPLOT to produce postscript
# ! CALL PGSLW(1)                                            ! select line weight
# ! CALL PGSCH(0.9)                                          ! select character height
# ! CALL PGENV(PGxMIN,PGxMAX,PGyMIN,PGyMAX,0,0)              ! construct frame
# ! CALL PGLAB('log\d10\u[\gl/\gmm]','     log\d10\u[\gx/cm\u2\dg\u-1\d]  and  10\fia',&
     # ! &'DUST EXTINCTION OPACITY, \gx, AND ALBEDO, \fia\fn, AS A FUNCTION OF WAVELENGTH, \gl.')
# ! CALL PGSLS(1)                                            ! select full line
# ! CALL PGLINE(WLlTOT,PGx,PGy)                              ! plot extinction curve
# ! CALL PGSLS(2)                                            ! select dashed line
# ! CALL PGLINE(WLlTOT,PGx,PGz)                              ! plot 10 x albedo
# ! PGx(1)=-1.00; PGx(2)=+0.00                               ! set limiting abscissae of lines
# ! PGy(1)=-2.70; PGy(2)=-2.70                               ! set limiting ordinates of extinction full-line legend
# ! PGz(1)=-3.80; PGz(2)=-3.80                               ! set limiting ordinates of albedo dashed-line legend
# ! CALL PGSLS(1)                                            ! select full line
# ! CALL PGLINE(2,PGx,PGy)                                   ! draw full line
# ! CALL PGTEXT(+0.30,-2.80,'DUST EXTINCTION OPACITY,')      ! print extinction ...
# ! CALL PGTEXT(+1.05,-3.20,'log\d10\u[\gx/cm\u2\dg\u-1\d]') ! ... full-line legend
# ! CALL PGSLS(2)                                            ! select dashed line
# ! CALL PGLINE(2,PGx,PGz)                                   ! draw dashed line
# ! CALL PGTEXT(+0.30,-3.90,'DUST ALBEDO, 10\fia\fn')        ! print albedo dashed-line legend
# ! CALL PGEND                                               ! close PGPLOT
# ! WRITE (*,*) ' '                                          ! print blank line

                                                         # ! ! [] PLANCK FUNCTIONS AND VOLUME EMISSIVITIES
# ! teT(1)=3.16; teT(2)=10.0; teT(3)=31.6; teT(4)=100.; teT(5)=316. ! input selected temperatures
# ! CALL PGBEG(0,'/XWINDOW',1,1)                             ! open PGPLOT to display on screen
# ! !CALL PGBEG(0,'/PS',1,2)                                  ! open PGPLOT to produce postscript
# ! CALL PGENV(0.2,4.6,-4.2,+0.6,0,0)                        ! construct frame
# ! CALL PGLAB('      log\d10\u[\gl/\gmm]','       log\d10\u[\fiB\fn\d\gl\u(\fiT\fn)]  and  log\d10\u[\fij\fn\d\gl\u(\fiT\fn)]',&
# ! &'PLANCK FUNCTION, \fiB\fn\d\gl\u(\fiT\fn), AND VOLUME EMISSIVITY, \fij\fn\d\gl\u(\fiT\fn), AS A FUNCTION OF WAVELENGTH, \gl. ')
# ! DO TEk=1,5                                               ! start loop over temperatures
  # ! ZZdumR=(0.143878E+05)/teT(TEk)                         !   compute lambda_T=hc/kT
  # ! WLlamMIN=0.03*ZZdumR                                   !   compute minimum significant wavelength
  # ! WLlamMAX=10.0*ZZdumR                                   !   compute maximum significant wavelength
  # ! PGyMAX=-0.1E+21                                        !   set PGyMAX to absurdly low value
  # ! PGzMAX=-0.1E+21                                        !   set PGzMAX to absurdly low value
  # ! PGy=-0.2E+21                                           !   set all PGy to even lower value
  # ! PGz=-0.2E+21                                           !   set all PGz to even lower value
  # ! DO WLl=1,WLlTOT                                        !   start loop over wavelengths
    # ! IF (WLlam(WLl)<WLlamMIN) CYCLE                       !     [IF] wavelength very low, [CYCLE]
    # ! IF (WLlam(WLl)>WLlamMAX) CYCLE                       !     [IF] wavelength very high, [CYCLE]
    # ! PGy(WLl)=1./                                        &!     compute ...........
           # ! &(WLlam(WLl)**5*(EXP(ZZdumR/WLlam(WLl))-1.))  !     ... Planck Function
    # ! PGz(WLl)=PGy(WLl)*WLchi(WLl)*(1.-WLalb(WLl))         !     compute volume emissivity
    # ! PGy(WLl)=LOG10(PGy(WLl))                             !     compute LOG(PGy)
    # ! PGz(WLl)=LOG10(PGz(WLl))                             !     compute LOG(PGz)
    # ! IF (PGy(WLl)>PGyMAX) PGyMAX=PGy(WLl)                 !     update PGyMAX, as appropriate
    # ! IF (PGz(WLl)>PGzMAX) PGzMAX=PGz(WLl)                 !     update PGzMAX, as appropriate
  # ! ENDDO                                                  !   end loop over wavelengths
  # ! PGy=PGy-PGyMAX                                         !   normalise PGy
  # ! PGz=PGz-PGzMAX                                         !   normalise PGz
  # ! CALL PGSLS(2)                                          !   invoke dashed line
  # ! CALL PGLINE(WLlTOT,PGx,PGy)                            !   plot Planck Function
  # ! CALL PGSLS(1)                                          !   invoke full line
  # ! CALL PGLINE(WLlTOT,PGx,PGz)                            !   plot volume emissivity
  # ! CALL PGTEXT(+0.85,+0.13,'316.K')                       !   label 316.K plots
  # ! CALL PGTEXT(+1.30,+0.13,'100.K')                       !   label 100.K plots
  # ! CALL PGTEXT(+1.75,+0.13,'31.6K')                       !   label 31.6K plots
  # ! CALL PGTEXT(+2.25,+0.13,'10.0K')                       !   label 10.0K plots
  # ! CALL PGTEXT(+2.75,+0.13,'3.16K')                       !   label 3.16K plots
  # ! CALL PGTEXT(+3.60,+0.13,'\fiB\fn\d\gl\u(\fiT\fn)')     !   print Planck-Function dashed-line legend
  # ! CALL PGTEXT(+3.62,-0.13,'\fij\fn\d\gl\u(\fiT\fn)')     !   print volume-emissivity full-line legend
  # ! PGx(1)=+3.96; PGx(2)=+4.34                             !   set limiting abscissae of lines
  # ! PGy(1)=+0.17; PGy(2)=+0.17                             !   set limiting ordinates of Planck-Function dashed-line
  # ! PGz(1)=-0.09; PGz(2)=-0.09                             !   set limiting ordinates of volume-emssivity full-line
  # ! CALL PGSLS(2)                                          !   select dashed line
  # ! CALL PGLINE(2,PGx,PGy)                                 !   draw Planck-Function dashed line
  # ! CALL PGSLS(1)                                          !   select full line
  # ! CALL PGLINE(2,PGx,PGz)                                 !   draw volume-emissivity full line
# ! ENDDO                                                    ! end loop over temperatures
# ! CALL PGEND                                               ! close PGPLOT
# ! WRITE (*,*) ' '                                          ! print blank line

# ! !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ! END SUBROUTINE RT_PlotDustProperties
# ! !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
