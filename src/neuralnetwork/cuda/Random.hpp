
#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

namespace neuralnetwork {
namespace cuda {

struct Random {
  curandState *d_state;
  unsigned numStates;

  __device__ void Initialise(unsigned threadIndex, unsigned seed) {
    curand_init(seed, threadIndex, 0, &d_state[threadIndex]);
  }

  __device__ float SampleUniform(unsigned threadIndex) {
    return curand_uniform(&d_state[threadIndex % numStates]);
  }

  static Random Create(unsigned numStates, unsigned seed);
  static void Cleanup(Random &rnd);
};
}
}
