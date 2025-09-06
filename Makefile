CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
NVFLAGS = -O3 -arch=sm_70

all: larkin_cuda_solver

larkin_cuda_solver: larkin_cuda_solver.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

clean:
	rm -f larkin_cuda_solver

