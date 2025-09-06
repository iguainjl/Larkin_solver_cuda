NVCC = nvcc
NVFLAGS = -O3 -arch=sm_75 -lcufft -lcurand

all: larkin_cuda_solver

larkin_cuda_solver: larkin_cuda_solver.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

clean:
	rm -f larkin_cuda_solver

