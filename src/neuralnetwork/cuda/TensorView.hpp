
#pragma once

#include "MatrixView.hpp"

namespace neuralnetwork {
namespace cuda {

struct TensorView {
  unsigned numLayers;
  MatrixView *layer;
};
}
}
