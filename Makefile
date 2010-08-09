SDK= $(CUDA_SDK)
CXXFLAGS+=-O3

INC_CUTIL= -I$(SDK)/C/common/inc
LD_CUTIL=-lcutil -L$(SDK)/C/lib -L$(SDK)/C/common/lib

TARGETS= hello hello_cuda fizzbuzz01 fizzbuzz02 cuda001 cuda002 smallpt_cpu smallpt_cuda

all: $(TARGETS)
clean:
	rm -f $(TARGETS)

hello: hello.cu
	nvcc -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $^

hello_cuda: hello_cuda.cu
	nvcc -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $^

fizzbuzz01: fizzbuzz01.cu
	nvcc -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $^

fizzbuzz02: fizzbuzz02.cu
	nvcc -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $^

smallpt_cpu: smallpt_cpu.cpp
	$(CXX) -o $@ $(CXFLAGS) -fopenmp $^

smallpt_cuda: smallpt_cuda.cu
	nvcc -o $@ $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $^

cuda001: cuda001.cu
	nvcc -o $@ $(LD_CUTIL) $(INC_CUTIL) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $^
cuda002: cuda002.cu
	nvcc -o $@ $(LD_CUTIL) $(INC_CUTIL) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $^

