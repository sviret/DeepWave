CXX = g++
LD  = g++
CFLAGS = -Wall -g -std=c++14

INCS = /usr/local/include/root/ $(PWD)/tclap-1.2.1/include/ .

%.o: %.cxx 
	@echo "*"
	@echo "* compile "$@
	@echo "*"
	$(CXX) $(CFLAGS) $(addprefix -I, $(INCS)) -c $< -o $@

og_ana:main.o testfft.o chirp.o chirpgen.o bankgen.o filtregen.o jobparams.o
	@echo "Build gravitational wave analysis tool"
	$(LD) $^ $(shell /usr/local/bin/root-config --libs) -lfftw3 -o $@

all : og_ana

clean: 
	\rm *.o	
