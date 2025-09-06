#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <getopt.h>
#include <curand.h>
#include <curand_kernel.h>

// =====================================================
// Nonlinear Larkin equation in 1D:
// ∂t h = ν ∂x ( (∂x h)^(2n-1) ) + η(x)
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

// ------------------- KERNELS -----------------------
__global__ void derivative_kernel(const float *h, float *dh, int N, float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int ip = (i+1) % N;
        int im = (i-1+N) % N;
        dh[i] = (h[ip] - h[im]) / (2.0f * dx);
    }
}

__global__ void flux_kernel(const float *u, float *F, int N, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = u[i];
        float absval = fabsf(val);
        F[i] = copysignf(powf(absval, 2*n-1), val);
    }
}

__global__ void update_kernel(float *h, const float *dFdx, const float *eta, int N, float dt, float nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        h[i] += dt * (nu * dFdx[i] + eta[i]);
    }
}

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

    if (tid == 0) {
        int sum = 0;
        for (int j=0;j<blockDim.x;j++) sum += local_count[j];
        atomicAdd(count,sum);
    }
}

// ----------------- MAIN ----------------------------
int main(int argc, char **argv) {
    // Default parameters
    int N = 8192;
    float L = 100.0f;
    int n = 2;
    float tmax = 2.0f;
    float dt = 0.001f;
    float nu = 1.0f;
    float Delta = 0.05f;
    unsigned int seed = 123;
    const char *outfile = "rho_vs_t.dat";

    // Command-line parsing
    static struct option long_options[] = {
        {"N", required_argument, 0, 'N'},
        {"L", required_argument, 0, 'L'},
        {"n", required_argument, 0, 'n'},
        {"tmax", required_argument, 0, 't'},
        {"dt", required_argument, 0, 'd'},
        {"Delta", required_argument, 0, 'D'},
        {"seed", required_argument, 0, 's'},
        {"out", required_argument, 0, 'o'},
        {0,0,0,0}
    };
    int opt, option_index=0;
    while((opt=getopt_long(argc,argv,"N:L:n:t:d:D:s:o:",long_options,&option_index))!=-1){
        switch(opt){
            case 'N': N = atoi(optarg); break;
            case 'L': L = atof(optarg); break;
            case 'n': n = atoi(optarg); break;
            case 't': tmax = atof(optarg); break;
            case 'd': dt = atof(optarg); break;
            case 'D': Delta = atof(optarg); break;
            case 's': seed = atoi(optarg); break;
            case 'o': outfile = optarg; break;
            default: break;
        }
    }

    float dx = L/N;

    // Device allocations
    float *d_h, *d_u, *d_F, *d_dFdx, *d_eta;
    gpuErrchk(cudaMalloc(&d_h,N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_u,N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_F,N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_dFdx,N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_eta,N*sizeof(float)));

    gpuErrchk(cudaMemset(d_h,0,N*sizeof(float)));

    // Host eta
    float *h_eta = (float*)malloc(N*sizeof(float));
    srand(seed);
    for(int i=0;i<N;i++){
        h_eta[i] = Delta * (2.0f*rand()/RAND_MAX -1.0f);
    }
    gpuErrchk(cudaMemcpy(d_eta,h_eta,N*sizeof(float),cudaMemcpyHostToDevice));

    // Time loop
    int steps = int(tmax/dt);
    dim3 block(256);
    dim3 grid((N+block.x-1)/block.x);

    // Prepare output file
    FILE *fout = fopen(outfile,"w");
    if(!fout){ fprintf(stderr,"Cannot open %s\n",outfile); return 1; }

    int *d_count;
    gpuErrchk(cudaMalloc(&d_count,sizeof(int)));

    for(int istep=0;istep<=steps;istep++){
        float t = istep*dt;

        derivative_kernel<<<grid,block>>>(d_h,d_u,N,dx);
        flux_kernel<<<grid,block>>>(d_u,d_F,N,n);
        derivative_kernel<<<grid,block>>>(d_F,d_dFdx,N,dx);
        update_kernel<<<grid,block>>>(d_h,d_dFdx,d_eta,N,dt,nu);

        if(istep%(steps/20)==0){ // every 5% print
            gpuErrchk(cudaMemset(d_count,0,sizeof(int)));
            count_zeros_kernel<<<grid,block>>>(d_u,N,d_count);
            int h_count=0;
            gpuErrchk(cudaMemcpy(&h_count,d_count,sizeof(int),cudaMemcpyDeviceToHost));
            float rho = float(h_count)/N;
            printf("t=%.3f rho=%.5f\n",t,rho);
            fprintf(fout,"%.6f %.8f\n",t,rho);
            fflush(fout);
        }
    }

    fclose(fout);
    cudaFree(d_h); cudaFree(d_u); cudaFree(d_F);
    cudaFree(d_dFdx); cudaFree(d_eta); cudaFree(d_count);
    free(h_eta);

    printf("Simulation finished, output written to %s\n",outfile);
    return 0;
}

