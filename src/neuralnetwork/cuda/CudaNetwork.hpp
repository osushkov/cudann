
#pragma once

#include "../../math/MatrixView.hpp"
#include "../NetworkSpec.hpp"
#include <memory>

namespace neuralnetwork {
namespace cuda {

class CudaNetwork {
public:
  CudaNetwork(const NetworkSpec &spec);
  ~CudaNetwork();

  void SetWeights(const std::vector<math::MatrixView> &weights);
  void GetWeights(std::vector<math::MatrixView> &outWeights);

  void Train(const math::MatrixView &batchInputs, const math::MatrixView &batchOutputs);

private:
  struct CudaNetworkImpl;
  std::unique_ptr<CudaNetworkImpl> impl;
};
}
}
