
#include "common/Common.hpp"
#include "math/Math.hpp"
#include "math/MatrixView.hpp"
#include "neuralnetwork/Network.hpp"
#include "neuralnetwork/cuda/CudaNetwork.hpp"

int main(int argc, char **argv) {
  neuralnetwork::NetworkSpec spec;
  spec.numInputs = 2;
  spec.numOutputs = 2;
  spec.hiddenLayers = {3};
  spec.maxBatchSize = 10;
  spec.hiddenActivation = neuralnetwork::LayerActivation::TANH;
  spec.outputActivation = neuralnetwork::LayerActivation::TANH;

  neuralnetwork::cuda::CudaNetwork::Initialise(spec);

  vector<EMatrix> weights;
  weights.push_back(EMatrix(3, 3));
  weights.push_back(EMatrix(2, 4));

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

  cout << "hello world" << endl;
  return 0;
}
