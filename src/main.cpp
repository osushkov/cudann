
#include "common/Common.hpp"
#include "math/Math.hpp"
#include "math/MatrixView.hpp"
#include "neuralnetwork/Network.hpp"
#include "neuralnetwork/cuda/CudaNetwork.hpp"

int main(int argc, char **argv) {
  neuralnetwork::NetworkSpec spec;
  spec.numInputs = 2;
  spec.numOutputs = 3;
  spec.hiddenLayers = {2};
  spec.maxBatchSize = 10;
  spec.hiddenActivation = neuralnetwork::LayerActivation::RELU;
  spec.outputActivation = neuralnetwork::LayerActivation::RELU;

  neuralnetwork::cuda::CudaNetwork::Initialise(spec);

  vector<EMatrix> weights;
  weights.push_back(EMatrix(2, 3));
  weights.push_back(EMatrix(3, 3));

  vector<math::MatrixView> views;
  for (auto &wm : weights) {
    // views.push_back(math::MatrixView::Create(wm.rows(), wm.cols()));
    wm.fill(0.0f);
    views.push_back(math::GetMatrixView(wm));
  }

  neuralnetwork::cuda::CudaNetwork::GetWeights(views);
  //
  // for (const auto &v : views) {
  //   unsigned i = 0;
  //   for (unsigned r = 0; r < v.rows; r++) {
  //     for (unsigned c = 0; c < v.cols; c++) {
  //       cout << v.data[i++] << endl;
  //     }
  //   }
  // }
  for (const auto &wm : weights) {
    cout << wm << endl << endl;
  }

  EMatrix input(3, 2);
  input(0, 0) = 1.0f;
  input(0, 1) = 2.0f;
  input(1, 0) = 3.0f;
  input(1, 1) = 4.0f;
  input(2, 0) = 5.0f;
  input(2, 1) = 6.0f;

  EMatrix output(2, 1);
  output(0, 0) = 10.0f;
  output(1, 0) = 12.0f;

  neuralnetwork::cuda::CudaNetwork::Train(math::GetMatrixView(input), math::GetMatrixView(output));

  cout << "hello world" << endl;
  return 0;
}
