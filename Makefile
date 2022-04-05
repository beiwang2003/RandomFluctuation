SYSTEMS = $(shell hostname)
COMPILER = gnu
CUDA_PATH = /usr/local/cuda-11.0

#CUDA_PATH should set in the calling shell if CMSSW tools are not used

#tigergpu at pinceton
ifneq (,$(findstring tigergpu, $(SYSTEMS)))
#git clone https://github.com/NVlabs/cub.git
	#CUBROOT=/home/beiwang/clustering/cub-1.8.0
	GPUARCH=sm_60
	CLHEP_PATH=/tigress/beiwang/local
endif 

ifeq ($(COMPILER), gnu)
	CC = g++
	CXXFLAGS += -std=c++11 -O3 -fopenmp -march=native \
	  -mprefer-vector-width=512 -fopt-info-vec -g \
	  -I$(CUDA_PATH)/include -I$(CLHEP_PATH)/include -DUSE_GPU -DGPU_TIMER
	LDFLAGS += -std=c++11 -O3 -fopenmp -march=native \
	  -mprefer-vector-width=512 -fopt-info-vec -g -lCLHEP-Random-2.4.5.1 -L$(CLHEP_PATH)/lib
endif

ifeq ($(COMPILER), intel)
	CC = icpc
	CXXFLAGS += -std=c++11 -O3 -qopenmp -xHost \
	  -qopt-zmm-usage=high -qopt-report=5 \
	  -I$(CUDA_PATH)/include -I$(CLHEP_PATH)/include -g -DUSE_GPU -DGPU_TIMER
	LDFLAGS += -std=c++11 -O3 -qopenmp -xHost \
	  -qopt-zmm-usage=high -qopt-report=5 -g -lCLHEP-Random-2.4.5.1 -L$(CLHEP_PATH)/lib
endif

NVCC = nvcc
CUDAFLAGS += -std=c++17 -O3 -g --default-stream per-thread -arch=$(GPUARCH) --ptxas-options=-v -lineinfo -use_fast_math #--maxrregcount 32
CUDALDFLAGS += -lcudart -L$(CUDA_PATH)/lib64

ifeq ($(COMPILER), intel)
	CUDAFLAGS += -ccbin=icpc #specify intel for nvcc host compiler
endif

landau_random: main.o SiG4UniversalFluctuation.o SiG4UniversalFluctuationGPU.o
	$(CC) $(LDFLAGS) $(CUDALDFLAGS) -o landau_random main.o SiG4UniversalFluctuation.o SiG4UniversalFluctuationGPU.o

main.o: main.cc
	$(CC) $(CXXFLAGS) -o main.o -c main.cc

SiG4UniversalFluctuation.o: SiG4UniversalFluctuation.cc 
	$(CC) $(CXXFLAGS) -o SiG4UniversalFluctuation.o -c SiG4UniversalFluctuation.cc

SiG4UniversalFluctuationGPU.o: SiG4UniversalFluctuationGPU.cu
	$(NVCC) $(CUDAFLAGS) -o SiG4UniversalFluctuationGPU.o -c SiG4UniversalFluctuationGPU.cu

clean:
	rm -rf landau_random *.o *.optrpt
