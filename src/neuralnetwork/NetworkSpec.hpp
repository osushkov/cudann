
#pragma once

#include <vector>

namespace neuralnetwork {

enum class LayerActivation { TANH, LOGISTIC, RELU, LEAKY_RELU, LINEAR, SOFTMAX };

struct NetworkSpec {
  unsigned numInputs;
  unsigned numOutputs;
  std::vector<unsigned> hiddenLayers;
  float nodeActivationRate;

  unsigned maxBatchSize;

  LayerActivation hiddenActivation;
  LayerActivation outputActivation;
};
}
