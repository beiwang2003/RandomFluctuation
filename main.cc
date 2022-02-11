#include "SiG4UniversalFluctuation.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include "CLHEP/Random/MixMaxRng.h"

#define SIZE 419922

int main(){


  SiG4UniversalFluctuation flucture;
  // make an engine object here
  CLHEP::MixMaxRng engine;

  // Generate charge fluctuations.
  float *sum = (float*)malloc(SIZE*sizeof(float));
  float *sum_fun = (float *)malloc(SIZE*sizeof(float));
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
    long seed[4];
    seed[0] = seed0[i];
    seed[1] = seed1[i];
    seed[2] = seed2[i];
    seed[3] = seed3[i];
    engine.setSeeds(seed, 4);
    const long *seed_test = engine.getSeeds();

    if (i<5) {
      std::cout<<mom_i<<" "<<particleMass_i<<" "<<deltaCutoff_i<<" "<<seglen_i<<" "<<segeloss_i<<std::endl;
      std::cout<<seed_test[0]<<" "<<seed_test[1]<<" "<<seed_test[2]<<" "<<seed_test[3]<<std::endl;
      std::cout<<NumberOfSegs<<" ";
    }

    for (int j = 0; j < NumberOfSegs; j++) {
      // The G4 routine needs momentum in MeV, mass in MeV, delta-cut in MeV,
    // track segment length in mm, segment eloss in MeV
    // Returns fluctuated eloss in MeV
    // the cutoff is sometimes redefined inside, so fix it.
      double tmp = flucture.SampleFluctuations(mom_i, particleMass_i, deltaCutoff_i, seglen_i, segeloss_i, &engine) / 1000.;
      sum_i += flucture.SampleFluctuations(mom_i, particleMass_i, deltaCutoff_i, seglen_i, segeloss_i, &engine) / 1000.;
      if (i<5) {
	std::cout<<tmp;
      }
    }

    if (i<5) std::cout<<std::endl;
    sum_fun[i] = sum_i;

  }
  /*
  for (int i=0; i<SIZE; i++) {
    if (i%10000==0)
      std::cout<<"i="<<i<<"numSegs"<<numSegs[i]<<"sum="<<sum[i]<<"sum_fun="<<sum_fun[i]<<std::endl;
  }
  */

  free(sum);
  free(sum_fun);
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
