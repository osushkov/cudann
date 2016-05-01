
#include "Random.hpp"
#include "Util.hpp"
#include <cassert>

using namespace neuralnetwork::cuda;

__global__ void setupStates(Random rnd, unsigned seed) {
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
    rnd.Initialise(id, seed);
}

Random Random::Create(unsigned numStates, unsigned seed) {
  assert(numStates > 0);

  Random rnd;
  rnd.numStates = numStates;
  cudaError_t err = cudaMalloc(&rnd.d_state, numStates * sizeof(curandState));
  CheckError(err);

  int tpb = 32;
  int bpg = (numStates + tpb - 1) / tpb;
  setupStates<<<bpg, tpb>>>(rnd, seed);

  return rnd;
}

void Random::Cleanup(Random &rnd) {
  cudaError_t err = cudaFree(rnd.d_state);
  CheckError(err);
  rnd.d_state = nullptr;
}
