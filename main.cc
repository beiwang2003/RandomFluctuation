#include "SiG4UniversalFluctuation.h"
#include <iostream>
#include <fstream>
#include <cassert>

#define SIZE 419922

int main(){


  SiG4UniversalFluctuation flucture;
  // make an engine object here
  // ????

  // Generate charge fluctuations.
  float *sum = (float*)malloc(SIZE*sizeof(float));
  float *sum_fun = (float *)malloc(SIZE*sizeof(float));
  double *deltaCutoff = (double*)malloc(SIZE*sizeof(double));
  double *mom = (double*)malloc(SIZE*sizeof(double));
  double *seglen = (double*)malloc(SIZE*sizeof(double));
  double *segeloss = (double*)malloc(SIZE*sizeof(double));
  double *particleMass = (double*)malloc(SIZE*sizeof(double));
  int *numSegs = (int *)malloc(SIZE*sizeof(int));

  std::ifstream input("data.txt");

  for (int i=0; i<SIZE; i++) {
    int NumnberOfSegs;

    input >> numSegs[i];
    input >> mom[i];
    input >> particleMass[i];
    input >> deltaCutoff[i];
    input >> seglen[i];
    input >> segeloss[i];

    input >> NumberOfSegs;
    assert(NumberOfSegs == numSegs[i]);

    sum[i] = 0.0;
    for (int j=0; j<NumberOfSegs; j++) {
      double tmp;

      input >> tmp;
      sum[i] += tmp;
    }
    sum_fun[i] = 0.0;
  }
  /*
  for (int i=0; i<SIZE; i++) {

    if (i%1000==0)
      std::cout<<"i="<<i<<"numSegs"<<numSegs[i]<<"sum="<<sum[i]<<"mom="<<mom[i]<<std::endl;
  }
  */

  for (int i=0; i<SIZE; i++) {
    float sum_i = 0.0;
    double mom_i = mom[i];
    double particleMass_i = particleMass[i];
    double deltaCutoff_i = deltaCutoff[i];
    double seglen_i = seglen[i];
    double segeloss_i = segeloss[i];

    int NumberOfSegs = numSegs[i];
    for (int j = 0; j < NumberOfSegs; j++) {
      // The G4 routine needs momentum in MeV, mass in MeV, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV
    // Returns fluctuated eloss in MeV
    // the cutoff is sometimes redefined inside, so fix it.

      sum_i += fluctuate->SampleFluctuations(mom_i, particleMass_i, deltaCutoff_i, seglen_i, segeloss_i, engine) / 1000.;
    }
    sum_fun[i] = sum_i;

  }

  free(sum);
  free(sum_fun);
  free(deltaCutoff);
  free(mom);
  free(particleMass);
  free(seglen);
  free(segeloss);
  free(numSegs);

  return 0;
}
