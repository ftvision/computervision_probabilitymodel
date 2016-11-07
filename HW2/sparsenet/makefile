MEX = /Applications/MATLAB_R2013b.app/bin/mex
NRFDIR = ./nrf
MEXT = mexmaci64

cgf.$(MEXT): cgf.c
	$(MEX) -I$(NRFDIR) -L$(NRFDIR) -lnrfopt cgf.c
