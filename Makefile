CC= gcc
CXX= g++

TOP= /usr/local/cuda
SDK= $(CUDA_SDK)

CFLAGS+=-g -O3 -Wall
CXXFLAGS+=-g -O3 -Wall
NVCCFLAGS+=-g -G -O3

INC_CUBLAS= -I$(TOP)/include
LD_CUBLAS= -lcublas -L$(TOP)/lib

INC_CUTIL= -I$(SDK)/C/common/inc
LD_CUTIL=-lcutil -L$(SDK)/C/lib -L$(SDK)/C/common/lib

TARGETS= hello hello_cuda fizzbuzz01 fizzbuzz02 white2black white2black_cuda cuda001 cuda002 smallpt_cpu smallpt_cuda ao ao_cuda

all: $(TARGETS)
clean:
	rm -f $(TARGETS)

hello: hello.cu
	nvcc -o $@ $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) $^

hello_cuda: hello_cuda.cu
	nvcc -o $@ $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) $^

fizzbuzz01: fizzbuzz01.cu
	nvcc -o $@ $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) $^

fizzbuzz02: fizzbuzz02.cu
	nvcc -o $@ $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) $^

white2black: white2black.c
	$(CC) -o $@ $(CFLAGS) -fopenmp $^

white2black_cuda: white2black_cuda.cu
	nvcc -o $@ $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) $^

cuda001: cuda001.cu
	nvcc -o $@ $(LD_CUTIL) $(INC_CUTIL) $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) $^
cuda002: cuda002.cu
	nvcc -o $@ $(LD_CUTIL) $(INC_CUTIL) $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) $^

ao: ao.c
	$(CC) -o $@ -lm $(CFLAGS) -fopenmp $^

ao_cuda: ao_cuda.cu
	nvcc -o $@ $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) $^

smallpt_cpu: smallpt_cpu.cpp
	$(CXX) -o $@ $(CXXFLAGS) -fopenmp $^

smallpt_cuda: smallpt_cuda.cu
	nvcc -o $@ $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) $^
