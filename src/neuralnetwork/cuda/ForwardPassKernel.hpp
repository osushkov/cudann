#pragma once

#include "../NetworkSpec.hpp"
#include "Types.hpp"

namespace neuralnetwork {
namespace cuda {
namespace ForwardPassKernel {

void Apply(LayerWeights layerWeights, LayerBatchOutputs input, LayerBatchOutputs output,
           LayerActivation activation);
}
}
}
