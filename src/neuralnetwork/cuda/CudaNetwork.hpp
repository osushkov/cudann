
#pragma once

// #include "../../math/Tensor.hpp"
#include "../../math/MatrixView.hpp"
#include "../NetworkSpec.hpp"

namespace neuralnetwork {
namespace cuda {
namespace CudaNetwork {

void Initialise(const NetworkSpec &spec);
void Cleanup(void);

void SetWeights(const std::vector<math::MatrixView> &weights);
void GetWeights(std::vector<math::MatrixView> &outWeights);

void Train(const math::MatrixView &batchInputs, const math::MatrixView &batchOutputs);
}
}
}
