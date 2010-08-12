CC= gcc
CXX= g++

TOP= /usr/local/cuda
SDK= $(CUDA_SDK)

INC=
CFLAGS+=-O3 -Wall
CXXFLAGS+=-O3 -Wall
NVCCFLAGS+=-O3

INC_CUBLAS= -I$(TOP)/include
LD_CUBLAS= -lcublas -L$(TOP)/lib

INC_CUTIL= -I$(SDK)/C/common/inc
LD_CUTIL=-lcutil_i386 -L$(SDK)/C/lib -L$(SDK)/C/common/lib

TARGETS= hello hello_cuda fizzbuzz01 fizzbuzz02 white2black white2black_cuda smallpt_cpu ao ao_cuda ao_cuda_opt

all: $(TARGETS)
clean:
	rm -f $(TARGETS)

hello: hello.cu
	nvcc -o $@ $(NVCCFLAGS) $(INC) $(LDFLAGS) $^

hello_cuda: hello_cuda.cu
	nvcc -o $@ $(NVCCFLAGS) $(INC) $(LDFLAGS) $^

fizzbuzz01: fizzbuzz01.cu
	nvcc -o $@ $(NVCCFLAGS) $(INC) $(LDFLAGS) $^

fizzbuzz02: fizzbuzz02.cu
	nvcc -o $@ $(NVCCFLAGS) $(INC) $(LDFLAGS) $^

white2black: white2black.c
	$(CC) -o $@ $(CFLAGS) -fopenmp $^

white2black_cuda: white2black_cuda.cu
	nvcc -o $@ $(NVCCFLAGS) $(INC) $(LDFLAGS) $^

ao: ao.c
	$(CC) -o $@ -lm $(CFLAGS) -fopenmp $^

ao_cuda: ao_cuda.cu
	nvcc -o $@ $(INC_CUTIL) $(NVCCFLAGS) $(INC) $(LDFLAGS) $^

ao_cuda_opt: ao_cuda_opt.cu
	nvcc -o $@ -use_fast_math $(INC_CUTIL) $(NVCCFLAGS) $(INC) $(LDFLAGS) $^

smallpt_cpu: smallpt_cpu.cpp
	$(CXX) -o $@ $(CXXFLAGS) -fopenmp $^
