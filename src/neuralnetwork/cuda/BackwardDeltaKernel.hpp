#pragma once

#include "Types.hpp"

namespace neuralnetwork {
namespace cuda {
namespace BackwardDeltaKernel {

void Apply(LayerBatchDeltas nextDelta, LayerWeights transposedWeights,
           LayerBatchOutputs layerOutput, LayerBatchDeltas outDelta, cudaStream_t stream);
}
}
}
