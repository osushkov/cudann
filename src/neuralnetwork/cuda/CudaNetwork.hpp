
#pragma once

// #include "../../math/Tensor.hpp"
#include "../NetworkSpec.hpp"
#include "MatrixView.hpp"
#include "TensorView.hpp"

namespace neuralnetwork {
namespace cuda {
namespace CudaNetwork {

void Initialise(const NetworkSpec &spec);

void SetWeights(const TensorView *weights);
void GetWeights(TensorView *out);

void Train(const MatrixView *batchInputs, const MatrixView *batchOutputs);
}
}
}
