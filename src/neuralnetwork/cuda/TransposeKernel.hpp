#pragma once

#include "Types.hpp"

namespace neuralnetwork {
namespace cuda {
namespace TransposeKernel {

void Apply(LayerWeights layerWeights, LayerWeights transposedWeights, cudaStream_t stream);
}
}
}
