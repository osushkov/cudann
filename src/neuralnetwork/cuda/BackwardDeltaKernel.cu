
#include "BackwardDeltaKernel.hpp"
#include "Constants.hpp"
#include <cuda_runtime.h>

using namespace neuralnetwork;
using namespace neuralnetwork::cuda;

void BackwardDeltaKernel::Apply(LayerBatchDeltas nextDelta, LayerWeights transposedWeights,
                                LayerBatchDeltas outDelta) {

}
