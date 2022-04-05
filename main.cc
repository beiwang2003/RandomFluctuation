#include "SiG4UniversalFluctuation.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include "CLHEP/Random/MixMaxRng.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#if _OPENMP
#include <omp.h>
#endif

#ifdef USE_GPU
#include "SiG4UniversalFluctuationGPU.cuh"
#endif

#define SIZE 419922

#ifdef USE_GPU
void gpu_timer_start(gpu_timing_t *gpu_timing, cudaStream_t stream) {
  CUDA_RT_CALL(cudaEventCreate(&gpu_timing->start));
  CUDA_RT_CALL(cudaEventCreate(&gpu_timing->stop));
  CUDA_RT_CALL(cudaEventRecord(gpu_timing->start, stream));
}

float gpu_timer_measure(gpu_timing_t *gpu_timing, cudaStream_t stream) {
  float elapsedTime;
  CUDA_RT_CALL(cudaEventRecord(gpu_timing->stop, stream));
  CUDA_RT_CALL(cudaEventSynchronize(gpu_timing->stop));
  CUDA_RT_CALL(cudaEventElapsedTime(&elapsedTime, gpu_timing->start, gpu_timing->stop));
  CUDA_RT_CALL(cudaEventRecord(gpu_timing->start, stream));

  return elapsedTime/1000;
}

float gpu_timer_measure_end(gpu_timing_t *gpu_timing, cudaStream_t stream) {
  float elapsedTime;
  CUDA_RT_CALL(cudaEventRecord(gpu_timing->stop,stream));
  CUDA_RT_CALL(cudaEventSynchronize(gpu_timing->stop));
  CUDA_RT_CALL(cudaEventElapsedTime(&elapsedTime, gpu_timing->start,gpu_timing->stop));

  CUDA_RT_CALL(cudaEventDestroy(gpu_timing->start));
  CUDA_RT_CALL(cudaEventDestroy(gpu_timing->stop));
  return elapsedTime/1000;
}
#endif

int main(){

  SiG4UniversalFluctuation flucture;
  // make an engine object here
  CLHEP::MixMaxRng engine;

  // Generate charge fluctuations
  double *deltaCutoff = (double*)malloc(SIZE*sizeof(double));
  double *mom = (double*)malloc(SIZE*sizeof(double));
  double *seglen = (double*)malloc(SIZE*sizeof(double));
  double *segeloss = (double*)malloc(SIZE*sizeof(double));
  double *particleMass = (double*)malloc(SIZE*sizeof(double));
  int *numSegs = (int *)malloc(SIZE*sizeof(int));
  long *seed0 = (long *)malloc(SIZE*sizeof(long));
  long *seed1 = (long *)malloc(SIZE*sizeof(long));
  long *seed2 = (long *)malloc(SIZE*sizeof(long));
  long *seed3 = (long *)malloc(SIZE*sizeof(long));

  std::ifstream input("data2.txt");

  int maxNumSegs = 0;
  int index=0;
  for (int i=0; i<SIZE; i++) {
    int NumberOfSegs;
    input >> seed0[i];
    input >> seed1[i];
    input >> seed2[i];
    input >> seed3[i];

    input >> numSegs[i];
    input >> mom[i];
    input >> particleMass[i];
    input >> deltaCutoff[i];
    input >> seglen[i];
    input >> segeloss[i];
    input >> NumberOfSegs;
    assert(NumberOfSegs == numSegs[i]);

    if (NumberOfSegs > maxNumSegs) {
      maxNumSegs = NumberOfSegs;
      index = i;
    }

    for (int j=0; j<NumberOfSegs; j++) {
      double tmp;
      input >> tmp;
    }
  }
  std::cout<<"maxNumSegs="<<maxNumSegs<<"at Index="<<index<<std::endl;

  double *fluctSum = (double *)malloc(SIZE * sizeof(double));
  double *fluctSum_h = (double *)malloc(SIZE * sizeof(double));
  for (int i=0; i<SIZE; i++) {
    fluctSum[i] = 0.0;
    fluctSum_h[i] = 0.0;
  }

#ifdef USE_GPU
  double t0 = omp_get_wtime();

  gpu_timing_t *gpu_timing = (gpu_timing_t *)malloc(sizeof(gpu_timing_t));

#ifdef GPU_TIMER
  gpu_timer_start(gpu_timing, 0);
#endif
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread. In curand_kernel.h, you will find:
     typedef struct curandStateXORWOW curandState_t, the default XORWOW generator */
  curandState_t *states;

  /* allocate space on the GPU for the random states */
  CUDA_RT_CALL(cudaMalloc((void**) &states, SIZE * sizeof(curandState_t)));

  /* allocate and initialize array variables for later use in GPU kernel */
  int *numSegs_d;
  long *seed_d;
  double *deltaCutoff_d, *mom_d, *seglen_d, *segeloss_d, *particleMass_d;
  double *fluctSum_d;
  CUDA_RT_CALL(cudaMalloc((void **)&numSegs_d, SIZE *sizeof(int)));
  CUDA_RT_CALL(cudaMalloc((void **)&seed_d, SIZE * sizeof(long)));
  CUDA_RT_CALL(cudaMalloc((void **)&deltaCutoff_d, SIZE * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc((void **)&mom_d, SIZE * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc((void **)&seglen_d, SIZE * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc((void **)&segeloss_d, SIZE * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc((void **)&particleMass_d, SIZE * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc((void **)&fluctSum_d, SIZE * sizeof(double)));

#ifdef GPU_TIMER
  gpu_timing->memAllocTime = gpu_timer_measure(gpu_timing, 0);
#endif

  CUDA_RT_CALL(cudaMemcpy(numSegs_d, numSegs, SIZE*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(seed_d, seed0, SIZE*sizeof(long), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(mom_d, mom, SIZE*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(particleMass_d, particleMass, SIZE*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(deltaCutoff_d, deltaCutoff, SIZE*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(seglen_d, seglen, SIZE*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(segeloss_d, segeloss, SIZE*sizeof(double), cudaMemcpyHostToDevice));

  initConstMemory(keV, proton_mass_c2, eV, electron_mass_c2, twopi_mc2_rcl2);

#ifdef GPU_TIMER
  gpu_timing->memTransDHTime = gpu_timer_measure(gpu_timing, 0);
#endif

  int nthreads = 256;
  int nblocks = (SIZE + nthreads - 1)/nthreads;

  getSamples(seed_d, numSegs_d, mom_d, particleMass_d, deltaCutoff_d, seglen_d, segeloss_d, states, fluctSum_d, SIZE);

#ifdef GPU_TIMER
  gpu_timing->kernelTime = gpu_timer_measure(gpu_timing, 0);
#endif

  CUDA_RT_CALL(cudaMemcpy(fluctSum_h, fluctSum_d, SIZE * sizeof(double), cudaMemcpyDeviceToHost));

#ifdef GPU_TIMER
  gpu_timing->memTransHDTime = gpu_timer_measure(gpu_timing, 0);
#endif

  cudaFree(states);
  cudaFree(deltaCutoff_d);
  cudaFree(mom_d);
  cudaFree(particleMass_d);
  cudaFree(seglen_d);
  cudaFree(segeloss_d);
  cudaFree(numSegs_d);
  cudaFree(seed_d);
  cudaFree(fluctSum_d);

#ifdef GPU_TIMER
  gpu_timing->memFreeTime = gpu_timer_measure_end(gpu_timing, 0);

  std::cout<<" GPU Memory Transfer Host to Device Time: "<<gpu_timing->memTransHDTime<<std::endl;
  std::cout<<" GPU Memory Transfer Device to Host Time: "<<gpu_timing->memTransDHTime<<std::endl;
  std::cout<<" GPU Memory Allocation Time: "<<gpu_timing->memAllocTime<<std::endl;
  std::cout<<" GPU Memory Free Time: "<<gpu_timing->memFreeTime<<std::endl;
  std::cout<<" GPU Kernel Time "<<gpu_timing->kernelTime<<std::endl;
#endif

  free(gpu_timing);
  double t1 = omp_get_wtime();

  std::cout<<"GPU random number generator time: "<<t1-t0<<std::endl;
#else
  double t0 = omp_get_wtime();

  for (int i=0; i<SIZE; i++) {
    double mom_i = mom[i];
    double particleMass_i = particleMass[i];
    double deltaCutoff_i = deltaCutoff[i];
    double seglen_i = seglen[i];
    double segeloss_i = segeloss[i];

    int NumberOfSegs = numSegs[i];
    long seed[4];
    seed[0] = seed0[i];
    seed[1] = seed1[i];
    seed[2] = seed2[i];
    seed[3] = seed3[i];
    //    engine.setSeeds(seed, 4);
    //const long *seed_test = engine.getSeeds();
    engine.setSeed(seed[0]);
    //    long seed_test = engine.getSeed();

    double sum = 0.0;
    for (int j = 0; j < NumberOfSegs; j++) {
      // The G4 routine needs momentum in MeV, mass in MeV, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV
    // Returns fluctuated eloss in MeV
    // the cutoff is sometimes redefined inside, so fix it.
      sum  +=  flucture.SampleFluctuations(mom_i, particleMass_i, deltaCutoff_i, seglen_i, segeloss_i, &engine) / 1000.;
    }
    fluctSum[i] = sum;
  }

  double t1 = omp_get_wtime();

  std::cout<<"CPU random number generator time: "<<t1-t0<<std::endl;
#endif

  free(fluctSum);
  free(fluctSum_h);
  free(deltaCutoff);
  free(mom);
  free(particleMass);
  free(seglen);
  free(segeloss);
  free(numSegs);
  free(seed0);
  free(seed1);
  free(seed2);
  free(seed3);

  return 0;
}
