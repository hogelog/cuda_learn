SDK= $(CUDA_SDK)
INCLUDES+= -I$(SDK)/C/common/inc
LDFLAGS+=-L$(SDK)/C/lib -L$(SDK)/C/common/lib

TARGETS= hello cuda001 cuda002

all: $(TARGETS)

hello: hello.cu
	nvcc -o $@ $(INCLUDES) $(LDFLAGS) $^

cuda001: cuda001.cu
	nvcc -o $@ -lcutil $(INCLUDES) $(LDFLAGS) $^
cuda002: cuda002.cu
	nvcc -o $@ -lcutil $(INCLUDES) $(LDFLAGS) $^


