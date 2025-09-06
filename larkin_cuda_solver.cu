// larkin_cuda_solver.cu
// CUDA spectral solver for nonlinear Larkin equation (single-precision)
// ∂_t h = nu ∂_x[ (∂_x h)^(2n-1) ] + η(x)   (quenched noise)
// Outputs:
//  - rho file: t  rho(t)  log(t)  log(rho)
//  - Su file:  k  Su(k)
//
// Command-line options (long):
//  --N, --L, --n, --tmax, --dt, --nu, --Delta, --seed, --nrec, --out, --outSu, --eps_factor
//
// Build: nvcc ... -lcufft -lcurand
// (use multi-gencode flags for portability)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <getopt.h>
#include <vector>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cufft.h>
#include <curand.h>

#define CHECK_CUDA(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); } } while(0)

#define CHECK_CUFFT(call) do { \
  cufftResult r = (call); \
  if (r != CUFFT_SUCCESS) { \
    fprintf(stderr,"CUFFT error %s:%d: %d\n", __FILE__, __LINE__, (int)r); exi

