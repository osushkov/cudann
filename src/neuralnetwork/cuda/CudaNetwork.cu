
#include "CudaNetwork.hpp"
#include "Util.hpp"

#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdio>

using namespace neuralnetwork;
using namespace neuralnetwork::cuda;
using namespace std;

static NetworkSpec networkSpec;
static vector<LayerWeights> d_layerWeights;
static vector<LayerBatchOutputs> d_layerOutputs;
static vector<LayerBatchDeltas> d_layerDeltas;
static SamplesBatch d_samplesBatch;

// Pre-allocated all of the device memory we will need. We should never have to malloc device
// memory after this function is called.
static void allocDeviceMemory(void) {
  vector<unsigned> layerSizes(networkSpec.hiddenLayers.size() + 1);
  for (unsigned i = 0; i < networkSpec.hiddenLayers.size(); i++) {
    layerSizes[i] = networkSpec.hiddenLayers[i];
  }
  layerSizes[networkSpec.hiddenLayers.size()] = networkSpec.numOutputs;

  // This is for the input layer
  d_layerOutputs.push_back(
      util::NewLayerBatchOutputs(networkSpec.maxBatchSize, networkSpec.numInputs + 1));

  for (unsigned i = 0; i < layerSizes.size(); i++) {
    unsigned prevLayerSize = i == 0 ? networkSpec.numInputs : layerSizes[i-1];

    d_layerWeights.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
    d_layerOutputs.push_back(util::NewLayerBatchOutputs(networkSpec.maxBatchSize, layerSizes[i] + 1));
    d_layerDeltas.push_back(util::NewLayerBatchDeltas(networkSpec.maxBatchSize, layerSizes[i]));
  }

  d_samplesBatch = util::NewSamplesBatch(networkSpec.maxBatchSize, networkSpec.numInputs);
}

static void freeDeviceMemory(void) {
  for (auto& lw : d_layerWeights) { util::DeleteLayerWeights(lw); }
  for (auto& lo : d_layerOutputs) { util::DeleteLayerBatchOutputs(lo); }
  for (auto& ld : d_layerDeltas) { util::DeleteLayerBatchDeltas(ld); }
  util::DeleteSamplesBatch(d_samplesBatch);
}

__global__ void initialiseLayer(LayerWeights layer, const float initRange) {
  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= layer.layerSize || col >= layer.inputSize) {
    return;
  }

  float *out = layer.Elem(row, col);
  *out = 1.123f;

  printf("woo: %x\n", out);
  printf("hello %d %d : %d %d = %f\n", row, col, layer.layerSize, layer.inputSize, *layer.Elem(row, col));
}

static void initialiseWeights(void) {
  int tpbX = 16;
  int tpbY = 16;

  for (auto& lw : d_layerWeights) {
    // Blocks per grid in X and Y dimensions.
    int bpgX = (lw.inputSize + tpbX - 1) / tpbX;
    int bpgY = (lw.layerSize + tpbY - 1) / tpbY;

    float initRange = 1.0f / sqrtf(lw.inputSize);

    cout << "dim: " << lw.layerSize << " " << lw.inputSize << endl;
    initialiseLayer<<<dim3(bpgX, bpgY, 1), dim3(tpbX, tpbY, 1)>>>(lw, initRange);
  }
}

static void initialiseOutputs(void) {
  for (auto& lo : d_layerOutputs) {

  }
}

void CudaNetwork::Initialise(const NetworkSpec &spec) {
  networkSpec = spec;
  allocDeviceMemory();
  initialiseWeights();
  initialiseOutputs();
}

void CudaNetwork::Cleanup(void) {
  freeDeviceMemory();
}

void CudaNetwork::SetWeights(const std::vector<math::MatrixView> &weights) {
  assert(d_layerWeights.size() == weights.size());

  for (unsigned i = 0; i < weights.size(); i++) {
    assert(weights[i].rows == d_layerWeights[i].layerSize);
    assert(weights[i].cols == d_layerWeights[i].inputSize);

    cudaError_t err = cudaMemcpy2D(
        d_layerWeights[i].weights, d_layerWeights[i].pitch,
        weights[i].data, weights[i].cols * sizeof(float),
        weights[i].cols, weights[i].rows,
        cudaMemcpyHostToDevice);

    CheckError(err);
  }
}

void CudaNetwork::GetWeights(std::vector<math::MatrixView> &outWeights) {
  assert(outWeights.size() == d_layerWeights.size());

  for (unsigned i = 0; i < outWeights.size(); i++) {
    std::cout << d_layerWeights[i].layerSize << " : " << outWeights[i].rows << std::endl;
    assert(outWeights[i].rows == d_layerWeights[i].layerSize);
    assert(outWeights[i].cols == d_layerWeights[i].inputSize);

    printf("get weights: %x\n", d_layerWeights[i].weights);
    // cudaError_t err = cudaMemcpy(
    //     outWeights[i].data, d_layerWeights[i].weights,
    //     outWeights[i].cols * outWeights[i].rows * sizeof(float),
    //     cudaMemcpyDeviceToHost);

    cudaError_t err = cudaMemcpy2D(
        outWeights[i].data, outWeights[i].cols * sizeof(float), // dst
        d_layerWeights[i].weights, d_layerWeights[i].pitch, // src
        outWeights[i].cols * sizeof(float), outWeights[i].rows, // width, height
        cudaMemcpyDeviceToHost);

    CheckError(err);
  }
}

void CudaNetwork::Train(const math::MatrixView &batchInputs, const math::MatrixView &batchOutputs) {

}
