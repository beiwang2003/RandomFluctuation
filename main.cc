#include "SiG4UniversalFluctuation.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include "CLHEP/Random/MixMaxRng.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#ifdef USE_GPU
#include "SiG4UniversalFluctuationGPU.cuh"
#endif

#define SIZE 419922

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

    if (NumberOfSegs > maxNumSegs) maxNumSegs = NumberOfSegs;

    for (int j=0; j<NumberOfSegs; j++) {
      double tmp;
      input >> tmp;
    }
  }
  double *fluct = (double *)malloc(SIZE * maxNumSegs * sizeof(double));
  for (int i=0; i<SIZE * maxNumSegs; i++) fluct[i] = 0.0;

#ifdef USE_GPU
  // reference: http://ianfinlayson.net/class/cpsc425/notes/cuda-random
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread */
  curandState_t *states;

  /* allocate space on the GPU for the random states */
  CUDA_RT_CALL(cudaMalloc((void**) &states, SIZE * sizeof(curandState_t)));

  /* allocate and initialize array variables for later use in GPU kernel */
  int *numSegs_d;
  long *seed_d;
  double *deltaCutoff_d, *mom_d, *seglen_d, *segeloss_d, *particleMass_d;
  double *fluct_d;
  CUDA_RT_CALL(cudaMalloc((void **)&numSegs_d, SIZE *sizeof(int)));
  CUDA_RT_CALL(cudaMalloc((void **)&seed_d, SIZE * sizeof(long)));
  CUDA_RT_CALL(cudaMalloc((void **)&deltaCutoff_d, SIZE * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc((void **)&mom_d, SIZE * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc((void **)&seglen_d, SIZE * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc((void **)&segeloss_d, SIZE * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc((void **)&particleMass_d, SIZE * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc((void **)&fluct_d, SIZE * maxNumSegs * sizeof(double)));

  CUDA_RT_CALL(cudaMemcpy(numSegs_d, numSegs, SIZE*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(seed_d, seed0, SIZE*sizeof(long), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(mom_d, mom, SIZE*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(particleMass_d, particleMass, SIZE*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(deltaCutoff_d, deltaCutoff, SIZE*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(seglen_d, seglen, SIZE*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpy(segeloss_d, segeloss, SIZE*sizeof(double), cudaMemcpyHostToDevice));

  /* initilize constant variables for later use in GPU kernel */
  // initialize constant variables on the host
  double chargeSquare_h = 1.0;

  // data members to speed up the fluctuation calculation
  double ipotFluct_h = 0.0001736;
  double electronDensity_h = 6.797E+20;

  double f1Fluct_h = 0.8571;
  double f2Fluct_h = 0.1429;
  double e1Fluct_h = 0.000116;
  double e2Fluct_h =  0.00196;
  double rateFluct_h = 0.4;
  double e1LogFluct_h = -9.063;
  double e2LogFluct_h = -6.235;
  double ipotLogFluct_h = -8.659;
  double e0_h = 1.E-5;

  double minNumberInteractionsBohr_h = 10.0;
  double theBohrBeta2_h = 50.0 * keV / proton_mass_c2;
  double minLoss_h = 10.0 * eV;
  double problim_h = 5.e-3;
  double sumalim_h = -log(problim_h);
  double alim_h = 10.;
  double nmaxCont1_h = 4.;
  double nmaxCont2_h = 16.;
  double electron_mass_c2_h = electron_mass_c2;
  double twopi_mc2_rcl2_h = twopi_mc2_rcl2;

  CUDA_RT_CALL(cudaMemcpyToSymbol(&chargeSquare, &chargeSquare_h, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&ipotFluct,&ipotFluct_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&electronDensity,&electronDensity_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&f1Fluct,&f1Fluct_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&f2Fluct,&f2Fluct_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&e1Fluct,&e1Fluct_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&e2Fluct,&e2Fluct_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&rateFluct,&rateFluct_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&e1LogFluct,&e1LogFluct_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&e2LogFluct,&e2LogFluct_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&ipotLogFluct,&ipotLogFluct_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&e0,&e0_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&minNumberInteractionsBohr_h,&minNumberInteractionsBohr_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&theBohrBeta2,&theBohrBeta2_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&minLoss,&minLoss_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&problim,&problim_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&sumalim,&sumalim_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&nmaxCont1,&nmaxCont1_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&nmaxCont2,&nmaxCont2_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&electron_mass_c2_d,&electron_mass_c2_h,sizeof(double),0, cudaMemcpyHostToDevice));
  CUDA_RT_CALL(cudaMemcpyToSymbol(&twopi_mc2_rcl2_d,&twopi_mc2_rcl2_h,sizeof(double),0, cudaMemcpyHostToDevice));

  int nthreads = 256;
  int nblocks = (SIZE + nthreads - 1)/nthreads;

  getSamples(seed_d, numSegs_d, mom_d, particleMass_d, deltaCutoff_d, seglen_d, segeloss_d, states, fluct_d, SIZE);

#else
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

    for (int j = 0; j < NumberOfSegs; j++) {
      // The G4 routine needs momentum in MeV, mass in MeV, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV
    // Returns fluctuated eloss in MeV
    // the cutoff is sometimes redefined inside, so fix it.
      fluct[i*maxNumSegs + j] = flucture.SampleFluctuations(mom_i, particleMass_i, deltaCutoff_i, seglen_i, segeloss_i, &engine) / 1000.;
    }

  }
#endif

#ifdef USE_GPU
  cudaFree(deltaCutoff_d);
  cudaFree(mom_d);
  cudaFree(particleMass_d);
  cudaFree(seglen_d);
  cudaFree(segeloss_d);
  cudaFree(numSegs_d);
  cudaFree(seed_d);
  cudaFree(fluct_d);
#endif

  free(fluct);
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
