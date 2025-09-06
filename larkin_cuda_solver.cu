#include <cstdio>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <cufft.h>

// =====================================================
// Nonlinear Larkin equation solver in CUDA
// ∂t h = ν ∂x ( (∂x h)^(2n-1) ) + η(x)
// with quenched noise η(x)
// =====================================================

// CUDA error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel for computing spatial derivative via finite difference
__global__ void derivative_kernel(const float *h, float *dh, int N, float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int ip = (i+1) % N;
        int im = (i-1+N) % N;
        dh[i] = (h[ip] - h[im]) / (2.0f * dx);
    }
}

// Kernel for nonlinear flux term F = sign(u) |u|^(2n-1)
__global__ void flux_kernel(const float *u, float *F, int N, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = u[i];
        float absval = fabsf(val);
        float powv = powf(absval, 2*n - 1);
        F[i] = copysignf(powv, val);
    }
}

// Euler step update
__global__ void update_kernel(float *h, const float *dFdx, const float *eta,
                              int N, float dt, float nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        h[i] += dt * (nu * dFdx[i] + eta[i]);
    }
}

// Count zero crossings in u = ∂x h
__global__ void count_zeros_kernel(const float *u, int N, int *count) {
    __shared__ int local_count[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    local_count[tid] = 0;

    if (i < N-1) {
        float s1 = u[i];
        float s2 = u[i+1];
        if (s1 * s2 < 0) local_count[tid] = 1;
    }
    __syncthreads();

    // reduce
    if (tid == 0) {
        int sum = 0;
        for (int j=0; j<blockDim.x; j++) sum += local_count[j];
        atomicAdd(count, sum);
    }
}

int main(int argc, char **argv) {
    // Parameters (could be parsed from argv)
    int N = 8192;
    float L = 100.0f;
    float dx = L / N;
    float dt = 0.001f;
    float tmax = 2.0f;
    float nu = 1.0f;
    int n = 2;

    // Allocate device arrays
    float *d_h, *d_u, *d_F, *d_dFdx, *d_eta;
    gpuErrchk(cudaMalloc(&d_h, N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_u, N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_F, N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_dFdx, N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_eta, N*sizeof(float)));

    // Initialize
    gpuErrchk(cudaMemset(d_h, 0, N*sizeof(float)));

    // Generate quenched disorder eta(x) on host
    float *h_eta = (float*)malloc(N*sizeof(float));
    for (int i=0; i<N; i++) {
        h_eta[i] = (rand() / (float)RAND_MAX - 0.5f); // uniform noise
    }
    gpuErrchk(cudaMemcpy(d_eta, h_eta, N*sizeof(float), cudaMemcpyHostToDevice));

    // Time loop
    int steps = int(tmax/dt);
    dim3 block(256);
    dim3 grid((N+block.x-1)/block.x);

    for (int step=0; step<steps; step++) {
        derivative_kernel<<<grid,block>>>(d_h, d_u, N, dx);
        flux_kernel<<<grid,block>>>(d_u, d_F, N, n);
        derivative_kernel<<<grid,block>>>(d_F, d_dFdx, N, dx);
        update_kernel<<<grid,block>>>(d_h, d_dFdx, d_eta, N, dt, nu);
    }

    // Final zero crossing count
    int *d_count, h_count=0;
    gpuErrchk(cudaMalloc(&d_count, sizeof(int)));
    gpuErrchk(cudaMemset(d_count, 0, sizeof(int)));
    count_zeros_kernel<<<grid,block>>>(d_u, N, d_count);
    gpuErrchk(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Zero crossings = %d\\n", h_count);

    // Cleanup
    cudaFree(d_h); cudaFree(d_u); cudaFree(d_F);
    cudaFree(d_dFdx); cudaFree(d_eta); cudaFree(d_count);
    free(h_eta);

    return 0;
}

