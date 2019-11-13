#
# Title:              pytest_unitTest.sh
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
#             Run with "bash pytest_unitTest.sh"
#             -v => verbose error messages
#             --tb=long => long traceback messages
#             --exitfirst => exit at first error
#
# Dependancies:
#             Python 3.x
#             Pytest
#                   and subsequent dependancies

pytest unitTest.py -v --tb=long --exitfirst
