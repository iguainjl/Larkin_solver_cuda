#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <getopt.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#define BLOCK_SIZE 256

// CUDA error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel: compute derivative u = ∂x h
__global__ void compute_u(const double *h, double *u, double dx, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int ip = (idx+1) % N;
        int im = (idx-1+N) % N;
        u[idx] = (h[ip] - h[im]) / (2.0*dx);
    }
}

// Kernel: compute nonlinear flux F(u) = sign(u) * (|u|+eps)^(2n-1)
__global__ void compute_flux(const double *u, double *F, int n, double eps, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double val = u[idx];
        double s = (val > 0) - (val < 0);
        F[idx] = s * pow(fabs(val)+eps, 2*n - 1);
    }
}

// Kernel: derivative of flux
__global__ void compute_dFdx(const double *F, double *dFdx, double dx, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int ip = (idx+1) % N;
        int im = (idx-1+N) % N;
        dFdx[idx] = (F[ip] - F[im]) / (2.0*dx);
    }
}

// Kernel: Euler update with quenched noise
__global__ void euler_update(double *h, const double *dFdx, const double *eta, double dt, double nu, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        h[idx] += dt * (nu*dFdx[idx] + eta[idx]);
    }
}

// Kernel: count zero crossings of u
__global__ void count_zeros(const double *u, int *counts, int N) {
    __shared__ int local[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int val = 0;
    if (idx < N) {
        int ip = (idx+1) % N;
        if (u[idx] == 0.0 || u[ip] == 0.0) val = 1;
        else if ((u[idx] > 0 && u[ip] < 0) || (u[idx] < 0 && u[ip] > 0)) val = 1;
    }
    local[tid] = val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s>>=1) {
        if (tid < s) local[tid] += local[tid+s];
        __syncthreads();
    }
    if (tid==0) counts[blockIdx.x] = local[0];
}

// Host: sum zero crossings
int get_zero_crossings(const double *u_d, int N) {
    int blocks = (N + BLOCK_SIZE -1)/BLOCK_SIZE;
    int *d_counts, *h_counts = new int[blocks];
    gpuErrchk(cudaMalloc(&d_counts, blocks*sizeof(int)));
    count_zeros<<<blocks,BLOCK_SIZE>>>(u_d,d_counts,N);
    gpuErrchk(cudaMemcpy(h_counts,d_counts,blocks*sizeof(int),cudaMemcpyDeviceToHost));
    int total =0;
    for(int i=0;i<blocks;i++) total += h_counts[i];
    cudaFree(d_counts);
    delete[] h_counts;
    return total;
}

// FFT-based structure factor
void compute_structure_factor(double *u_d, int N, double L, const std::string &fname) {
    cufftHandle plan;
    cufftDoubleReal *data_d;
    cufftDoubleComplex *fft_d;
    int Nc = N/2 + 1;

    gpuErrchk(cudaMalloc(&data_d, N*sizeof(double)));
    gpuErrchk(cudaMemcpy(data_d, u_d, N*sizeof(double), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMalloc(&fft_d, Nc*sizeof(cufftDoubleComplex)));

    cufftPlan1d(&plan,N,CUFFT_D2Z,1);
    cufftExecD2Z(plan, data_d, fft_d);

    std::vector<double> Su(Nc);
    cufftDoubleComplex *fft_h = new cufftDoubleComplex[Nc];
    gpuErrchk(cudaMemcpy(fft_h, fft_d, Nc*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
    for(int k=0;k<Nc;k++){
        double re=fft_h[k].x;
        double im=fft_h[k].y;
        Su[k] = (re*re + im*im)/N;
    }

    std::ofstream fout(fname);
    double dk = 2*M_PI/L;
    for(int k=0;k<Nc;k++){
        fout << k*dk << " " << Su[k] << "\n";
    }
    fout.close();
    cufftDestroy(plan);
    cudaFree(data_d); cudaFree(fft_d);
    delete[] fft_h;
}

// ---------------- MAIN ------------------
int main(int argc, char **argv) {
    // Defaults
    int N=1024, n=2, nout=100;
    double L=100.0, tmax=10.0, dt=0.01, nu=1.0, Delta=0.05, epsfactor=1e-8;
    double tmin=1e-6;
    std::string rhoFile="rho_vs_t.dat", SuFile="Su_final.dat";
    unsigned long seed=1234;

    static struct option long_options[] = {
        {"N", required_argument,0,'N'},
        {"L", required_argument,0,'L'},
        {"n", required_argument,0,'n'},
        {"tmax", required_argument,0,'t'},
        {"dt", required_argument,0,'d'},
        {"nu", required_argument,0,'u'},
        {"Delta", required_argument,0,'D'},
        {"epsfactor", required_argument,0,'e'},
        {"seed", required_argument,0,'s'},
        {"nout", required_argument,0,'o'},
        {"out", required_argument,0,'r'},
        {"outSu", required_argument,0,'S'},
        {0,0,0,0}
    };
    int opt, idx;
    while((opt=getopt_long(argc,argv,"",long_options,&idx))!=-1){
        switch(opt){
            case 'N': N=atoi(optarg); break;
            case 'L': L=atof(optarg); break;
            case 'n': n=atoi(optarg); break;
            case 't': tmax=atof(optarg); break;
            case 'd': dt=atof(optarg); break;
            case 'u': nu=atof(optarg); break;
            case 'D': Delta=atof(optarg); break;
            case 'e': epsfactor=atof(optarg); break;
            case 's': seed=atol(optarg); break;
            case 'o': nout=atoi(optarg); break;
            case 'r': rhoFile=optarg; break;
            case 'S': SuFile=optarg; break;
        }
    }

    double dx = L/N;
    int steps = int(tmax/dt);

    // Device memory
    double *h_d,*u_d,*F_d,*dFdx_d,*eta_d;
    gpuErrchk(cudaMalloc(&h_d,N*sizeof(double)));
    gpuErrchk(cudaMalloc(&u_d,N*sizeof(double)));
    gpuErrchk(cudaMalloc(&F_d,N*sizeof(double)));
    gpuErrchk(cudaMalloc(&dFdx_d,N*sizeof(double)));
    gpuErrchk(cudaMalloc(&eta_d,N*sizeof(double)));
    gpuErrchk(cudaMemset(h_d,0,N*sizeof(double)));

    // Initialize quenched noise
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,seed);
    curandGenerateNormalDouble(gen,eta_d,N,0.0,Delta);
    // subtract mean
    std::vector<double> eta_h(N);
    gpuErrchk(cudaMemcpy(eta_h.data(),eta_d,N*sizeof(double),cudaMemcpyDeviceToHost));
    double mean=0.0;
    for(int i=0;i<N;i++) mean+=eta_h[i];
    mean/=N;
    for(int i=0;i<N;i++) eta_h[i]-=mean;
    gpuErrchk(cudaMemcpy(eta_d,eta_h.data(),N*sizeof(double),cudaMemcpyHostToDevice));
    curandDestroyGenerator(gen);

    int blocks = (N+BLOCK_SIZE-1)/BLOCK_SIZE;

    // Output times (log spaced)
    std::vector<double> t_out(nout);
    double r = pow(tmax/tmin,1.0/(nout-1));
    t_out[0]=tmin;
    for(int i=1;i<nout;i++) t_out[i]=t_out[i-1]*r;

    std::ofstream fout(rhoFile);
    int next_out=0;

    for(int step=0;step<=steps;step++){
        double t=step*dt;

        // compute derivatives
        compute_u<<<blocks,BLOCK_SIZE>>>(h_d,u_d,dx,N);
        compute_flux<<<blocks,BLOCK_SIZE>>>(u_d,F_d,n,epsfactor,N);
        compute_dFdx<<<blocks,BLOCK_SIZE>>>(F_d,dFdx_d,dx,N);
        euler_update<<<blocks,BLOCK_SIZE>>>(h_d,dFdx_d,eta_d,dt,nu,N);

        // output
        if(next_out<nout && t>=t_out[next_out]){
            int zeros = get_zero_crossings(u_d,N);
            double rho = zeros/L;
            fout << t << " " << rho << "\n";
            next_out++;
        }
    }
    fout.close();

    // structure factor
    compute_u<<<blocks,BLOCK_SIZE>>>(h_d,u_d,dx,N); // final u
    compute_structure_factor(u_d,N,L,SuFile);

    cudaFree(h_d); cudaFree(u_d); cudaFree(F_d); cudaFree(dFdx_d); cudaFree(eta_d);

    printf("Simulation done. Output: %s, %s\n",rhoFile.c_str(),SuFile.c_str());
    return 0;
}

