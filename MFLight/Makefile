CXX = g++
LD  = g++
CFLAGS = -Wall -g -std=c++14

INCS = /opt/root-6.14.06/include/ $(PWD)/tclap-1.2.1/include/ .

%.o: %.cxx 
	@echo "*"
	@echo "* compile "$@
	@echo "*"
	$(CXX) $(CFLAGS) $(addprefix -I, $(INCS)) -c $< -o $@

og_ana:main.o chirp.o chirpgen.o bankgen.o filtregen.o jobparams.o
	@echo "Build gravitational wave analysis tool"
	$(LD) $^ $(shell /opt/root-6.14.06/bin/root-config --libs) -lfftw3 -o $@

all : og_ana

clean: 
	\rm *.o	
