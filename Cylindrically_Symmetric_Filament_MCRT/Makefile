FCOMP = gfortran
FCFLAGS = -O3 -march=native -flto
FCFLAGS = -O3 -freal-4-real-8 -march=native -flto
FCDEBUG = -g -fbacktrace -fcheck=all -fbounds-check -ffpe-trap=invalid,overflow,underflow,denormal
FCBUILD = -Wall -Wextra -pedantic
#FCOMP = ifort
#FCFLAGS = -O3 -ipo -xHOST
#FCDEBUG = -g -debug -traceback -check all
#FCBUILD = -warn all -stand f08

PROGRAM = run

### Put modules to be included here! ###
SRCS=		RadTrans_Subroutines.f90 \
						RadTrans_MainCode.F90

OBJECTS = $(SRCS:.f90=.o)

all:	$(PROGRAM)
debug:	FCFLAGS += $(FCDEBUG)
debug:	$(PROGRAM)
build:	FCFLAGS += $(FCBUILD)
build:	$(PROGRAM)

$(PROGRAM):	$(OBJECTS)
	$(FCOMP) $(FCFLAGS) -o $@ $^

%.o:  %.f90
	$(FCOMP)  $(FCFLAGS) -c $<

.PHONY:	clean

clean:
	rm -f *.o *.mod *.MOD run

### Courtesy of Dr. Lewis McMillan, St. Andrews ###
