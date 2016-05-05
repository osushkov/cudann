#pragma once

#include "Types.hpp"

namespace neuralnetwork {
namespace cuda {
namespace BackwardDeltaKernel {

void Apply(LayerBatchDeltas nextDelta, LayerWeights transposedWeights, LayerBatchDeltas outDelta);
}
}
}
