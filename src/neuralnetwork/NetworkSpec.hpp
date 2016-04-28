
#pragma once

#include <vector>

namespace neuralnetwork {

enum class LayerActivation { TANH, LOGISTIC, RELU, LEAKY_RELU, LINEAR, SOFTMAX };

struct NetworkSpec {
  unsigned numInputs;
  unsigned numOutputs;
  std::vector<unsigned> hiddenLayers;

  LayerActivation hiddenActivation;
  LayerActivation outputActivation;
};
}
