#include "GradientKernel.hpp"
#include "Constants.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

using namespace neuralnetwork;
using namespace neuralnetwork::cuda;

void GradientKernel::Apply(LayerBatchDeltas layerDeltas, LayerBatchOutputs layerOutputs,
                           LayerWeights outGradient) {

  assert(layerDeltas.batchSize == layerOutputs.batchSize);
  assert(layerDeltas.layerSize == outGradient.layerSize);
  assert(layerOutputs.layerSize == outGradient.inputSize);

  // int bpgX = (layerWeights.inputSize + TPB_X - 1) / TPB_X;
  // int bpgY = (layerWeights.layerSize + TPB_Y - 1) / TPB_Y;
}
