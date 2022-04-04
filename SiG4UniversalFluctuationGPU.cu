#include "SiG4UniversalFluctuationGPU.cuh"
#include <stdio.h>
#include <stdlib.h>

/* this GPU kernel function is used to initialize the random states */
__global__ void init_states(const long* seed_d, curandState_t* states, int SIZE) {

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < SIZE) {
  /* we have to initialize the state */
    curand_init(seed_d[gid],
		gid, /* the sequence number should be different for each thread */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&states[gid]);
  }
}

/* replace random number generator function calls from CLHEP with curand. For more info about random number generator from curand library, see: https://docs.nvidia.com/cuda/curand/device-api-overview.html#distributions
E.g.: replace CLHEP::RandFlat with curand_uniform_double (return double between 0.0-1.0)
      replace CLHEP::RandGaussQ with curand_log_normal_double (This function returns a double log-normally distributed float based on a normal distribution with the given mean and standard deviation)
      replace CLHEP::RandPoissonQ with curand_poisson (return unsigned int)
For more info about random number generator from curand library, see: https://docs.nvidia.com/cuda/curand/device-api-overview.html#distributions
      replace vdt::fast_log with log
Note: the return value is scaled with 0.001 to avoid /1000. operation outside the function call in the CPU version
*/
__global__ void sampleFluctuations_kernel(const int *numSegs_d,
					  const double *mom_d,
					  const double *particleMass_d,
					  const double *deltaCutoff_d,
					  const double *seglen_d,
					  const double *segeloss_d,
					  curandState_t *states,
					  double *fluct_d,
					  int SIZE) {
  // Calculate actual loss from the mean loss.
  // The model used to get the fluctuations is essentially the same
  // as in Glandz in Geant3 (Cern program library W5013, phys332).
  // L. Urban et al. NIM A362, p.416 (1995) and Geant4 Physics Reference Manual

  // shortcut for very very small loss (out of validity of the model)
  //
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < SIZE) {
    int numSegs = numSegs_d[gid];
    const double momentum = mom_d[gid];
    const double mass = particleMass_d[gid];
    const double tmax = deltaCutoff_d[gid];
    const double length = seglen_d[gid];
    const double meanLoss = segeloss_d[gid];
    curandState_t localState = states[gid];

    for (int i=0; i<numSegs; i++) {

      if (meanLoss < minLoss) {
	fluct_d[gid] = meanLoss*0.001;
	return;
      }

      double particleMass = mass;
      double gam2 = (momentum * momentum) / (particleMass * particleMass) + 1.0;
      double beta2 = 1.0 - 1.0 / gam2;
      double gam = sqrt(gam2);

      double loss(0.), siga(0.);

      // Gaussian regime
      // for heavy particles only and conditions
      // for Gauusian fluct. has been changed
      //
      if ((particleMass > electron_mass_c2_d) && (meanLoss >= minNumberInteractionsBohr * tmax)) {
	//std::cout << "if one\n";
	double massrate = electron_mass_c2_d / particleMass;
	double tmaxkine = 2. * electron_mass_c2_d * beta2 * gam2 / (1. + massrate * (2. * gam + massrate));
	if (tmaxkine <= 2. * tmax) {
	  siga = (1.0 / beta2 - 0.5) * twopi_mc2_rcl2_d * tmax * length * electronDensity * chargeSquare;
	  siga = sqrt(siga);
	  double twomeanLoss = meanLoss + meanLoss;
	  if (twomeanLoss < siga) {
	    double x;
	    do {
	      // bwang 04/04/22: replace CLHEP::RandFlat with curand_uniform_double
	      //loss = twomeanLoss * CLHEP::RandFlat::shoot(engine);
	      loss = twomeanLoss * curand_uniform_double(&localState);
	      //std::cout << "wml" << loss << " " << twomeanLoss << endl;
	      x = (loss - meanLoss) / siga;
	      //} while (1.0 - 0.5 * x * x < CLHEP::RandFlat::shoot(engine));
	    } while (1.0 - 0.5 * x * x < curand_uniform_double(&localState));
	  } else {
	    do {
	      // bwang 04/04/22: replace CLHEP::RandGaussQ with curand_log_normal_double
	      //loss = CLHEP::RandGaussQ::shoot(engine, meanLoss, siga);
	      loss = curand_log_normal_double(&localState, meanLoss, siga);
	    } while (loss < 0. || loss > twomeanLoss);
	  }
	  fluct_d[gid] = loss*0.001;
	  return;
	}
      }

      double a1 = 0., a2 = 0., a3 = 0.;
      double p3;
      double rate = rateFluct;

      double w1 = tmax / ipotFluct;
      // bwang 04/04/22: replace vdt::fast_log with log
      // double w2 = vdt::fast_log(2. * electron_mass_c2 * beta2 * gam2) - beta2;
      double w2 = log(2. * electron_mass_c2_d * beta2 * gam2) - beta2;
      //std::cout<< "w2 " << w2 << std::endl;
      if (w2 > ipotLogFluct) {
	double C = meanLoss * (1. - rateFluct) / (w2 - ipotLogFluct);
	a1 = C * f1Fluct * (w2 - e1LogFluct) / e1Fluct;
	a2 = C * f2Fluct * (w2 - e2LogFluct) / e2Fluct;
	if (a2 < 0.) {
	  a1 = 0.;
	  a2 = 0.;
	  rate = 1.;
	}
      } else {
	rate = 1.;
      }

      // added
      if (tmax > ipotFluct) {
	// bwang 04/04/22: replace vdt::fast_log with log
	//a3 = rate * meanLoss * (tmax - ipotFluct) / (ipotFluct * tmax * vdt::fast_log(w1));
	a3 = rate * meanLoss * (tmax - ipotFluct) / (ipotFluct * tmax * log(w1));
      }
      double suma = a1 + a2 + a3;
      //std::cout << "suma " << suma << std::endl;
      // Glandz regime
      //
      if (suma > sumalim) {
	if ((a1 + a2) > 0.) {
	  double p1, p2;
	  // excitation type 1
	  if (a1 > alim) {
	    siga = sqrt(a1);
	    // bwang 04/04/22: replace CLHEP::RandGaussQ with curand_log_normal_double
	    // p1 = max(0., CLHEP::RandGaussQ::shoot(engine, a1, siga) + 0.5);
	    p1 = max(0., curand_log_normal_double(&localState, a1, siga) + 0.5);
	  } else {
	    // bwang 04/04/22: replace CLHEP::RandPoissonQ with curand_poisson
	    //p1 = double(CLHEP::RandPoissonQ::shoot(engine, a1));
	    p1 = double(curand_poisson(&localState, a1));
	  }
	  //std::cout << "p1 "<< p1<< std::endl;
	  // excitation type 2
	  if (a2 > alim) {
	    siga = sqrt(a2);
	    // bwang 04/04/22: replace CLHEP::RandGaussQ with curand_log_normal_double
	    //p2 = max(0., CLHEP::RandGaussQ::shoot(engine, a2, siga) + 0.5);
	    p2 = max(0., curand_log_normal_double(&localState,a2, siga) + 0.5);
	  } else {
	    // bwang 04/04/22: replace CLHEP::RandPoissonQ with curand_poisson
	    //p2 = double(CLHEP::RandPoissonQ::shoot(engine, a2));
	    p2 = double(curand_poisson(&localState, a2));
	  }

	  loss = p1 * e1Fluct + p2 * e2Fluct;
	  //std::cout << "loss "<< loss<< std::endl;

	  // smearing to avoid unphysical peaks
	  if (p2 > 0.)
	    // bwang 04/04/22: replace CLHEP::RandFlat with curand_uniform_double
	    //loss += (1. - 2. * CLHEP::RandFlat::shoot(engine)) * e2Fluct;
	    loss += (1. - 2. * curand_uniform_double(&localState)) * e2Fluct;
	  else if (loss > 0.)
	    //loss += (1. - 2. * CLHEP::RandFlat::shoot(engine)) * e1Fluct;
	    loss += (1.- 2. * curand_uniform_double(&localState)) * e1Fluct;
	  if (loss < 0.)
	    loss = 0.0;
	}

	// ionisation
	if (a3 > 0.) {
	  if (a3 > alim) {
	    siga = sqrt(a3);
	    // bwang 04/04/22: replace CLHEP::RandGaussQ with curand_log_normal_double
	    //p3 = max(0., CLHEP::RandGaussQ::shoot(engine, a3, siga) + 0.5);
	    p3 = max(0., curand_log_normal_double(&localState,a3, siga) + 0.5);
	  } else {
	    // bwang 04/04/22: replace CLHEP::RandPoissonQ with curand_poisson
	    //p3 = double(CLHEP::RandPoissonQ::shoot(engine, a3));
	    p3 = double(curand_poisson(&localState, a3));
	  }
	  //std::cout << "p3 " << std::endl;
	  double lossc = 0.;
	  if (p3 > 0) {
	    double na = 0.;
	    double alfa = 1.;
	    if (p3 > nmaxCont2) {
	      double rfac = p3 / (nmaxCont2 + p3);
	      double namean = p3 * rfac;
	      double sa = nmaxCont1 * rfac;
	      // bwang 04/04/22: replace CLHEP::RandGaussQ with curand_log_normal_double
	      //na = CLHEP::RandGaussQ::shoot(engine, namean, sa);
	      na = curand_log_normal_double(&localState, namean, sa);
	      if (na > 0.) {
		alfa = w1 * (nmaxCont2 + p3) / (w1 * nmaxCont2 + p3);
		// bwang 04/04/22: replace vdt::fast_log with log
		// double alfa1 = alfa * vdt::fast_log(alfa) / (alfa - 1.);
		double alfa1 = alfa * log(alfa) / (alfa - 1.);
		double ea = na * ipotFluct * alfa1;
		double sea = ipotFluct * sqrt(na * (alfa - alfa1 * alfa1));
		//lossc += CLHEP::RandGaussQ::shoot(engine, ea, sea);
		lossc += curand_log_normal_double(&localState, ea, sea);
	      }
	    }

	    if (p3 > na) {
	      w2 = alfa * ipotFluct;
	      double w = (tmax - w2) / tmax;
	      int nb = int(p3 - na);
	      for (int k = 0; k < nb; k++)
		// bwang 04/04/22: replace CLHEP::RandFlat with curand_uniform_double
		//lossc += w2 / (1. - w * CLHEP::RandFlat::shoot(engine));
		lossc += w2 / (1. - w * curand_uniform_double(&localState));
	    }
	  }
	  loss += lossc;
	}

	fluct_d[gid] = loss * 0.001;
	return;
      }
      //std::cout << "verysmall \n";
      // suma < sumalim;  very small energy loss;
      // bwang 04/04/22: replace vdt::fast_log with log
      //a3 = meanLoss * (tmax - e0) / (tmax * e0 * vdt::fast_log(tmax / e0));
      a3 = meanLoss * (tmax - e0) / (tmax * e0 * log(tmax / e0));
      if (a3 > alim) {
	siga = sqrt(a3);
	// bwang 04/04/22: replace CLHEP::RandGaussQ with curand_log_normal_double
	//p3 = max(0., CLHEP::RandGaussQ::shoot(engine, a3, siga) + 0.5);
	p3 = max(0., curand_log_normal_double(&localState, a3, siga) + 0.5);
      } else {
	// bwang 04/04/22: replace CLHEP::RandPoissonQ with curand_poisson
	//p3 = double(CLHEP::RandPoissonQ::shoot(engine, a3));
	p3 = double(curand_poisson(&localState, a3));
      }
      if (p3 > 0.) {
	double w = (tmax - e0) / tmax;
	double corrfac = 1.;
	if (p3 > nmaxCont2) {
	  corrfac = p3 / nmaxCont2;
	  p3 = nmaxCont2;
	}
	int ip3 = (int)p3;
	for (int i = 0; i < ip3; i++)
	  // bwang 04/04/22: replace CLHEP::RandFlat with curand_uniform_double
	  //loss += 1. / (1. - w * CLHEP::RandFlat::shoot(engine));
	  loss += 1. / (1. - w * curand_uniform_double(&localState));
	loss *= e0 * corrfac;
	// smearing for losses near to e0
	if (p3 <= 2.)
	  //loss += e0 * (1. - 2. * CLHEP::RandFlat::shoot(engine));
	  loss += e0 * (1. - 2. * curand_uniform_double(&localState));
      }
      fluct_d[gid] = loss * 0.001;
      return;
    }
  }
}

  void getSamples(const long* seed_d,
                  const int *numSegs_d,
		  const double *mom_d,
		  const double *particleMass_d,
		  const double *deltaCutoff_d,
		  const double *seglen_d,
		  const double *segeloss_d,
		  curandState_t *states,
		  double *fluct_d,
		  int SIZE) {

    int nthreads = 256;
    int nblocks = (SIZE + nthreads - 1)/nthreads;

    init_states<<<nblocks, nthreads>>>(seed_d, states, SIZE);
    CUDA_RT_CALL(cudaGetLastError());

    sampleFluctuations_kernel<<<nblocks, nthreads>>>(numSegs_d, mom_d, particleMass_d, deltaCutoff_d, seglen_d, segeloss_d, states, fluct_d, SIZE);
    CUDA_RT_CALL(cudaGetLastError());

    cudaDeviceSynchronize();
  }
