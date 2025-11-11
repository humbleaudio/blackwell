NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
OUT_DIR = out

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = sudo $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
# Use -gencode to specify both compute and code architectures
# NVCC_FLAGS += -gencode arch=compute_90a,code=sm_90a
# NVCC_FLAGS += -gencode arch=compute_100,code=sm_100
NVCC_FLAGS += -gencode arch=compute_120,code=sm_120

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

sum: sum.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

sumprofile: sum
	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^

matmul: matmul.cu 
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

matmulprofile: matmul
	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^

clean:
	rm $(OUT_DIR)/*
