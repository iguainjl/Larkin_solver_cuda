// larkin_cuda_solver.cu
// CUDA spectral solver for nonlinear Larkin eq (single-precision)
// ∂_t h = nu ∂_x[ (∂_x h)^(2n-1) ] + η(x)   (quenched noise)
// Writes two-column files: rho(t) and Su(k)

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
    fprintf(stderr,"CUFFT error %s:%d: %d\n", __FILE__, __LINE__, (int)r); exit(EXIT_FAILURE);} } while(0)

#define CHECK_CURAND(call) do { \
  curandStatus_t s = (call); \
  if (s != CURAND_STATUS_SUCCESS) { \
    fprintf(stderr,"CURAND error %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(EXIT_FAILURE);} } while(0)

// ---------------- GPU kernels ----------------

// multiply complex spectrum H (Nc entries) by (i*k): H <- (i k) H
__global__ void multiply_by_ik(cufftComplex* H, const float* kpos, int Nc) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= Nc) return;
  float a = H[j].x;
  float b = H[j].y;
  float k = kpos[j];
  H[j].x = -k * b;
  H[j].y =  k * a;
}

// scale real array by scalar: a[i] *= s
__global__ void scale_array(float* a, float s, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) a[i] *= s;
}

// compute flux F = sign(u) * |u|^power, power is integer
__global__ void compute_flux(const float* u, float* F, int N, int power) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float ui = u[i];
  float absu = fabsf(ui);
  if (absu == 0.0f) { F[i] = 0.0f; return; }
  // powf is fine for single-precision
  float pv = powf(absu, (float)power);
  F[i] = (ui >= 0.0f) ? pv : -pv;
}

// update h: h += dt*(nu * dFdx + eta)
__global__ void update_h(float* h, const float* dFdx, const float* eta, float nu, float dt, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) h[i] += dt * (nu * dFdx[i] + eta[i]);
}

// ---------------- host helper: density counting (thresholded) ----------------
double density_from_host_u(const float* u_host, int N, double L) {
  // compute mean & std
  double mean = 0.0;
  for (int i=0;i<N;i++) mean += u_host[i];
  mean /= (double)N;
  double var = 0.0;
  for (int i=0;i<N;i++) { double d = u_host[i]-mean; var += d*d; }
  double std = sqrt(var / (double)N);
  double eps = 1e-3 * (std > 0.0 ? std : 1.0); // threshold

  // find initial sign (skip near-zero)
  int prev_sign = 0;
  for (int i=0;i<N;i++) {
    if (u_host[i] > eps) { prev_sign = 1; break; }
    if (u_host[i] < -eps) { prev_sign = -1; break; }
  }
  if (prev_sign == 0) return 0.0; // all ~0

  int changes = 0;
  for (int i=1;i<N;i++) {
    int s;
    if (u_host[i] > eps) s = 1;
    else if (u_host[i] < -eps) s = -1;
    else s = prev_sign; // treat small values as continuation
    if (s != prev_sign) { changes++; prev_sign = s; }
  }
  // wrap check between last and first
  int s0;
  if (u_host[0] > eps) s0 = 1; else if (u_host[0] < -eps) s0 = -1; else s0 = prev_sign;
  if (s0 != prev_sign) changes++;

  return ((double)changes) / L;
}

// logspace vector
std::vector<float> logspace(float tmin, float tmax, int nrec) {
  std::vector<float> v(nrec);
  double l0 = log(tmin), l1 = log(tmax);
  for (int i=0;i<nrec;i++) v[i] = (float)exp(l0 + (double)i/(nrec-1)*(l1-l0));
  return v;
}

// ---------------- main ----------------
int main(int argc, char** argv) {
  // defaults
  int N = 16384;
  double L = 100.0;
  int n = 2;
  double tmax = 50.0;
  double dt = -1.0; // choose default later
  double nu = 1.0;
  double Delta = 0.05;
  unsigned long long seed = 12345ull;
  int nrec = 200;
  const char* out_rho = "rho_vs_t.dat";
  const char* out_su  = "Su_final.dat";

  // parse options
  static struct option opts[] = {
    {"N", required_argument, 0, 0},
    {"L", required_argument, 0, 0},
    {"n", required_argument, 0, 0},
    {"tmax", required_argument, 0, 0},
    {"dt", required_argument, 0, 0},
    {"nu", required_argument, 0, 0},
    {"Delta", required_argument, 0, 0},
    {"seed", required_argument, 0, 0},
    {"nrec", required_argument, 0, 0},
    {"out", required_argument, 0, 0},
    {"outSu", required_argument, 0, 0},
    {0,0,0,0}
  };
  int optidx=0;
  while (1) {
    int c = getopt_long(argc, argv, "", opts, &optidx);
    if (c == -1) break;
    if (c==0) {
      const char* name = opts[optidx].name;
      if (!strcmp(name,"N")) N = atoi(optarg);
      else if (!strcmp(name,"L")) L = atof(optarg);
      else if (!strcmp(name,"n")) n = atoi(optarg);
      else if (!strcmp(name,"tmax")) tmax = atof(optarg);
      else if (!strcmp(name,"dt")) dt = atof(optarg);
      else if (!strcmp(name,"nu")) nu = atof(optarg);
      else if (!strcmp(name,"Delta")) Delta = atof(optarg);
      else if (!strcmp(name,"seed")) seed = strtoull(optarg,NULL,10);
      else if (!strcmp(name,"nrec")) nrec = atoi(optarg);
      else if (!strcmp(name,"out")) out_rho = optarg;
      else if (!strcmp(name,"outSu")) out_su = optarg;
    }
  }

  // device check
  int devCount=0;
  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  if (devCount==0) {
    fprintf(stderr,"No CUDA device found. Are you in a GPU runtime?\n");
    return 1;
  }
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop,0));
  printf("Using CUDA device 0: %s (compute %d.%d), %.2f GB\n",
         prop.name, prop.major, prop.minor, (double)prop.totalGlobalMem/(1024.0*1024.0*1024.0));

  // derived
  if (dt <= 0.0) {
    double dx = L / (double)N;
    dt = 0.02 * dx * dx; // stable-ish small dt
  }
  int nsteps = (int)ceil(tmax / dt);
  int Nc = N/2 + 1;
  double dx = L / (double)N;

  printf("Params: N=%d L=%.6g n=%d tmax=%.6g dt=%.6g nsteps=%d nu=%.6g Delta=%.6g seed=%llu\n",
         N, L, n, tmax, dt, nsteps, nu, Delta, seed);

  // allocate device arrays
  float *d_h=nullptr, *d_u=nullptr, *d_F=nullptr, *d_dFdx=nullptr, *d_eta=nullptr;
  cufftComplex *d_Hk=nullptr, *d_Fk=nullptr;
  float *d_kpos=nullptr, *d_Su=nullptr;

  CHECK_CUDA(cudaMalloc((void**)&d_h, sizeof(float)*N));
  CHECK_CUDA(cudaMalloc((void**)&d_u, sizeof(float)*N));
  CHECK_CUDA(cudaMalloc((void**)&d_F, sizeof(float)*N));
  CHECK_CUDA(cudaMalloc((void**)&d_dFdx, sizeof(float)*N));
  CHECK_CUDA(cudaMalloc((void**)&d_eta, sizeof(float)*N));
  CHECK_CUDA(cudaMalloc((void**)&d_Hk, sizeof(cufftComplex)*Nc));
  CHECK_CUDA(cudaMalloc((void**)&d_Fk, sizeof(cufftComplex)*Nc));
  CHECK_CUDA(cudaMalloc((void**)&d_kpos, sizeof(float)*Nc));
  CHECK_CUDA(cudaMalloc((void**)&d_Su, sizeof(float)*Nc));

  // prepare kpos on host then copy
  std::vector<float> h_kpos(Nc);
  for (int j=0;j<Nc;j++) h_kpos[j] = (float)((2.0*M_PI) * (double)j / L);
  CHECK_CUDA(cudaMemcpy(d_kpos, h_kpos.data(), sizeof(float)*Nc, cudaMemcpyHostToDevice));

  // set h = 0
  CHECK_CUDA(cudaMemset(d_h, 0, sizeof(float)*N));

  // create cuRAND generator and produce quenched normal noise on device
  curandGenerator_t gen;
  CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
  float sigma = (float)sqrt(Delta / dx);
  CHECK_CURAND(curandGenerateNormal(gen, d_eta, N, 0.0f, sigma));

  // copy eta to host, subtract mean, copy back (to remove k=0)
  std::vector<float> h_eta(N);
  CHECK_CUDA(cudaMemcpy(h_eta.data(), d_eta, sizeof(float)*N, cudaMemcpyDeviceToHost));
  double mean_eta = 0.0;
  for (int i=0;i<N;i++) mean_eta += h_eta[i];
  mean_eta /= (double)N;
  for (int i=0;i<N;i++) h_eta[i] -= (float)mean_eta;
  CHECK_CUDA(cudaMemcpy(d_eta, h_eta.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
  // destroy cuRAND generator
  CHECK_CURAND(curandDestroyGenerator(gen));

  // cuFFT plans
  cufftHandle plan_r2c_h, plan_c2r_h, plan_r2c_F, plan_c2r_F;
  CHECK_CUFFT(cufftPlan1d(&plan_r2c_h, N, CUFFT_R2C, 1));
  CHECK_CUFFT(cufftPlan1d(&plan_c2r_h, N, CUFFT_C2R, 1));
  CHECK_CUFFT(cufftPlan1d(&plan_r2c_F, N, CUFFT_R2C, 1));
  CHECK_CUFFT(cufftPlan1d(&plan_c2r_F, N, CUFFT_C2R, 1));

  // prepare recording times log-spaced between tmin and tmax
  float tmin = 1e-6f;
  std::vector<float> rec_times = logspace(tmin, (float)tmax, nrec);
  std::vector<float> rec_rho(nrec, 0.0f);

  // host buffer for u snapshot
  float *h_u = (float*)malloc(sizeof(float)*N);

  // kernel launch params
  int threads = 256;
  int blocksN = (N + threads - 1) / threads;
  int blocksNc = (Nc + threads - 1) / threads;
  int power = 2*n - 1;

  // open output file for rho(t)
  FILE *frho = fopen(out_rho, "w");
  if (!frho) { fprintf(stderr,"Cannot open %s for writing\n", out_rho); return 1; }
  fprintf(frho,"# t rho(t)\n");

  printf("Starting time loop (nsteps=%d)...\n", nsteps);
  int rec_idx = 0;
  double t = 0.0;
  for (int step=0; step < nsteps; ++step) {
    // spectral derivative: u = ∂_x h
    CHECK_CUFFT(cufftExecR2C(plan_r2c_h, d_h, d_Hk));
    multiply_by_ik<<<blocksNc, threads>>>(d_Hk, d_kpos, Nc);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUFFT(cufftExecC2R(plan_c2r_h, d_Hk, d_u));
    // normalize by N
    scale_array<<<blocksN, threads>>>(d_u, 1.0f/(float)N, N);

    // compute flux F = sign(u) * |u|^(2n-1)
    compute_flux<<<blocksN, threads>>>(d_u, d_F, N, power);
    CHECK_CUDA(cudaGetLastError());

    // spectral derivative of F to get dFdx
    CHECK_CUFFT(cufftExecR2C(plan_r2c_F, d_F, d_Fk));
    multiply_by_ik<<<blocksNc, threads>>>(d_Fk, d_kpos, Nc);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUFFT(cufftExecC2R(plan_c2r_F, d_Fk, d_dFdx));
    scale_array<<<blocksN, threads>>>(d_dFdx, 1.0f/(float)N, N);

    // Euler update for h
    update_h<<<blocksN, threads>>>(d_h, d_dFdx, d_eta, (float)nu, (float)dt, N);
    CHECK_CUDA(cudaGetLastError());

    t += dt;

    // record at scheduled times (log-spaced)
    while (rec_idx < nrec && t >= rec_times[rec_idx]) {
      // compute u again (ensure u up-to-date): do spectral derivative quickly
      CHECK_CUFFT(cufftExecR2C(plan_r2c_h, d_h, d_Hk));
      multiply_by_ik<<<blocksNc, threads>>>(d_Hk, d_kpos, Nc);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUFFT(cufftExecC2R(plan_c2r_h, d_Hk, d_u));
      scale_array<<<blocksN, threads>>>(d_u, 1.0f/(float)N, N);

      // copy u to host
      CHECK_CUDA(cudaMemcpy(h_u, d_u, sizeof(float)*N, cudaMemcpyDeviceToHost));

      double rho = density_from_host_u(h_u, N, L);
      rec_rho[rec_idx] = (float)rho;
      fprintf(frho, "%.8g %.12g\n", rec_times[rec_idx], rho);
      fflush(frho);
      rec_idx++;

      // progress print
      if (rec_idx % (nrec/10+1) == 0) {
        printf("Recorded %d/%d (t=%.6g, rho=%.6g)\n", rec_idx, nrec, rec_times[rec_idx-1], rho);
      }
    }
  }

  fclose(frho);
  printf("Finished time loop. Recorded %d samples in %s\n", rec_idx, out_rho);

  // compute final Su(k): forward transform of u
  CHECK_CUFFT(cufftExecR2C(plan_r2c_h, d_h, d_Hk));
  // Now Hk is i*k * H(h) — careful: we want spectrum of u. We want spectrum of u = ∂x h.
  // But above we multiplied by i*k in derivative steps; instead, compute u freshly:
  // recompute u properly:
  CHECK_CUFFT(cufftExecR2C(plan_r2c_h, d_h, d_Hk));
  multiply_by_ik<<<blocksNc, threads>>>(d_Hk, d_kpos, Nc);
  CHECK_CUDA(cudaGetLastError());
  // Now Hk contains spectrum of u (un-normalized). Copy Hk to host and compute Su.
  // Copy Hk to host
  std::vector<cufftComplex> h_Hk(Nc);
  CHECK_CUDA(cudaMemcpy(h_Hk.data(), d_Hk, sizeof(cufftComplex)*Nc, cudaMemcpyDeviceToHost));

  // compute Su on host and write to file
  FILE *fsu = fopen(out_su, "w");
  if (!fsu) { fprintf(stderr,"Cannot open %s for writing\n", out_su); return 1; }
  fprintf(fsu, "# k Su(k)\n");
  for (int j=1;j<Nc;j++) {
    float a = h_Hk[j].x;
    float b = h_Hk[j].y;
    double mag2 = (double)a*(double)a + (double)b*(double)b;
    // normalization to match Python code: Su = mag2 / N^2 * (N * dx)
    double Su = mag2 / ((double)N * (double)N) * ((double)N * dx);
    double k = (2.0*M_PI) * (double)j / L;
    fprintf(fsu, "%.12g %.12g\n", k, Su);
  }
  fclose(fsu);
  printf("Saved Su(k) to %s (positive k bins j=1..%d)\n", out_su, Nc-1);

  // cleanup
  free(h_u);
  CHECK_CUDA(cudaFree(d_h)); CHECK_CUDA(cudaFree(d_u));
  CHECK_CUDA(cudaFree(d_F)); CHECK_CUDA(cudaFree(d_dFdx));
  CHECK_CUDA(cudaFree(d_eta)); CHECK_CUDA(cudaFree(d_Hk)); CHECK_CUDA(cudaFree(d_Fk));
  CHECK_CUDA(cudaFree(d_kpos)); CHECK_CUDA(cudaFree(d_Su));
  CHECK_CUFFT(cufftDestroy(plan_r2c_h)); CHECK_CUFFT(cufftDestroy(plan_c2r_h));
  CHECK_CUFFT(cufftDestroy(plan_r2c_F)); CHECK_CUFFT(cufftDestroy(plan_c2r_F));

  printf("All done.\n");
  return 0;
}

