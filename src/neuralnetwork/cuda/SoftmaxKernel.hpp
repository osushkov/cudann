#pragma once

#include "Types.hpp"

namespace neuralnetwork {
namespace cuda {
namespace SoftmaxKernel {

void Apply(const LayerBatchOutputs &lastLayer, cudaStream_t stream);
}
}
}
