#pragma once

#include "Types.hpp"

namespace neuralnetwork {
namespace cuda {
namespace GradientKernel {

void Apply(LayerBatchDeltas layerDeltas, LayerBatchOutputs layerOutputs, LayerWeights outGradient,
           cudaStream_t stream);
}
}
}
