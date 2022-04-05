#ifndef _SIG4UNIVERSALFluctuationGPU_KERNEL_
#define _SIG4UNIVERSALFluctuationGPU_KERNEL_

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include "cuda_rt_call.h"
/*
__constant__ double chargeSquare;
__constant__ double ipotFluct;
__constant__ double electronDensity;

__constant__ double f1Fluct;
__constant__ double f2Fluct;
__constant__ double e1Fluct;
__constant__ double e2Fluct;
__constant__ double rateFluct;
__constant__ double e1LogFluct;
__constant__ double e2LogFluct;
__constant__ double ipotLogFluct;
__constant__ double e0;

__constant__ double minNumberInteractionsBohr;
__constant__ double theBohrBeta2;
__constant__ double minLoss;
__constant__ double problim;
__constant__ double sumalim;
__constant__ double alim;
__constant__ double nmaxCont1;
__constant__ double nmaxCont2;

__constant__ double electron_mass_c2_d;
__constant__ double twopi_mc2_rcl2_d;
*/
typedef struct {
  float memTransHDTime;
  float memTransDHTime;
  float memAllocTime;
  float memFreeTime;
  float kernelTime;
  cudaEvent_t start, stop;
} gpu_timing_t;

void getSamples(const long* seed_d,
                const int *numSegs_d,
                const double *mom_d,
	        const double *particleMass_d,
                const double *deltaCutoff_d,
                const double *seglen_d,
                const double *segeloss_d,
                curandState_t *states,
                double *fluctSum_d,
                int SIZE);

void initConstMemory(double keV, double proton_mass_c2, double eV, double electron_mass_c2, double twopi_mc2_rcl2);


#endif