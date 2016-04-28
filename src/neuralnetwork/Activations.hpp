
#pragma once

#include "NetworkSpec.hpp"
#include <cassert>
#include <cmath>

namespace neuralnetwork {

inline float ActivationValue(LayerActivation func, float in) {
  switch (func) {
  case LayerActivation::TANH:
    return tanhf(in);
  case LayerActivation::LOGISTIC:
    return 1.0f / (1.0f + expf(-in));
  case LayerActivation::RELU:
    return in > 0.0f ? in : 0.0f;
  case LayerActivation::LEAKY_RELU:
    return in > 0.0f ? in : (0.01f * in);
  case LayerActivation::LINEAR:
  case LayerActivation::SOFTMAX:
    return in;
  }
  assert(false);
  return in;
}

inline float ActivationDerivative(LayerActivation func, float in, float val) {
  switch (func) {
  case LayerActivation::TANH:
    return 1.0f - val * val;
  case LayerActivation::LOGISTIC:
    return val * (1.0f - val);
  case LayerActivation::RELU:
    return in > 0.0f ? 1.0f : 0.0f;
  case LayerActivation::LEAKY_RELU:
    return in > 0.0f ? 1.0f : 0.01f;
  case LayerActivation::LINEAR:
  case LayerActivation::SOFTMAX:
    return 1.0f;
  }
  assert(false);
  return 0.0f;
}
}
