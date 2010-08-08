SDK= $(CUDA_SDK)
INCLUDES+= -I$(SDK)/C/common/inc
LDFLAGS+=-L$(SDK)/C/lib -L$(SDK)/C/common/lib

TARGETS= hello hello_cuda fizzbuzz01 cuda001 cuda002

all: $(TARGETS)
clean:
	rm -f $(TARGETS)

hello: hello.cu
	nvcc -o $@ $(INCLUDES) $(LDFLAGS) $^

hello_cuda: hello_cuda.cu
	nvcc -o $@ $(INCLUDES) $(LDFLAGS) $^

fizzbuzz01: fizzbuzz01.cu
	nvcc -o $@ $(INCLUDES) $(LDFLAGS) $^

cuda001: cuda001.cu
	nvcc -o $@ -lcutil $(INCLUDES) $(LDFLAGS) $^
cuda002: cuda002.cu
	nvcc -o $@ -lcutil $(INCLUDES) $(LDFLAGS) $^


