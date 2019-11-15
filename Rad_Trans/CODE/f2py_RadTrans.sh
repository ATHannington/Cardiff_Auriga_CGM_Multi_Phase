#
# Title:              f2py_RadTrans.sh
# Created by:         Andrew T. Hannington
#
# Use with:           unitTest.py
#                         Created by: A. T. Hannington
#                     RadTrans_*.f90
#                         Created by: A. P. Whitworth et al.
#
# Date Created:       2019-11-13
#
# Usage Notes:
#             Run with "bash f2py_RadTrans.sh"
#             PLEASE NOTE: "python3" command is specific to my environment.
#                         Please change to whichever command uses python 3 in
#                         your environment.
#             F2Py essentially converts the f90 code to python functions. These
#             can then be utilised with
#             import RadTransSubroutinesf90 as f90Sub
#             f90sub.[subroutine name]
#
#
# Dependancies:
#               Python 3.x
#               NumPy
#                     and subsequent dependancies

python3 -m numpy.f2py -c RadTrans_Constants.f90 RadTrans_Subroutines.f90 -m RadTransSubroutinesf90 --fcompiler="gfortran" --f90flags="-g -fbounds-check -fbacktrace" --opt="-o3"
python3 -m numpy.f2py -c RadTrans_Constants.f90 -m RadTransConstantsf90 --fcompiler="gfortran" --f90flags="-g -fbounds-check -fbacktrace" --opt="-o3"
