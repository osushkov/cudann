#pragma once

#include "../NetworkSpec.hpp"
#include "Random.hpp"
#include "Types.hpp"

namespace neuralnetwork {
namespace cuda {
namespace ForwardPassKernel {

void Apply(LayerWeights layerWeights, LayerBatchOutputs input, LayerBatchOutputs output,
           LayerActivation activation, Random rnd, float nodeActivationRate, bool isOutputLayer,
           cudaStream_t stream);
}
}
}
