
#include "CudaNetwork.hpp"
#include "Util.hpp"
#include "Random.hpp"
#include "SoftmaxKernel.hpp"
#include "ForwardPassKernel.hpp"
#include "TransposeKernel.hpp"
#include "BackwardDeltaKernel.hpp"
#include "Constants.hpp"

#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdio>

#include <curand.h>
#include <cuda_runtime.h>

using namespace neuralnetwork;
using namespace neuralnetwork::cuda;
using namespace std;

static Random rnd;
static NetworkSpec networkSpec;
static vector<LayerWeights> d_layerWeights;
static vector<LayerWeights> d_layerGradients;
static vector<LayerBatchOutputs> d_layerOutputs;
static vector<LayerBatchDeltas> d_layerDeltas;
static SamplesBatch d_samplesBatch;

static LayerWeights d_transposeScratch;

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

  unsigned maxInputSize = 0;
  unsigned maxLayerSize = 0;

  for (unsigned i = 0; i < layerSizes.size(); i++) {
    unsigned prevLayerSize = i == 0 ? networkSpec.numInputs : layerSizes[i-1];

    maxInputSize = max(maxInputSize, prevLayerSize + 1);
    maxLayerSize = max(maxLayerSize, layerSizes[i]);

    d_layerWeights.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
    d_layerGradients.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
    d_layerOutputs.push_back(util::NewLayerBatchOutputs(networkSpec.maxBatchSize, layerSizes[i] + 1));
    d_layerDeltas.push_back(util::NewLayerBatchDeltas(networkSpec.maxBatchSize, layerSizes[i]));
  }

  d_samplesBatch =
      util::NewSamplesBatch(networkSpec.maxBatchSize, networkSpec.numInputs, networkSpec.numOutputs);

  d_transposeScratch = util::NewLayerWeights(maxLayerSize, maxInputSize);
}

static void freeDeviceMemory(void) {
  for (auto& lw : d_layerWeights) { util::DeleteLayerWeights(lw); }
  for (auto& lg : d_layerGradients) { util::DeleteLayerWeights(lg); }
  for (auto& lo : d_layerOutputs) { util::DeleteLayerBatchOutputs(lo); }
  for (auto& ld : d_layerDeltas) { util::DeleteLayerBatchDeltas(ld); }
  util::DeleteSamplesBatch(d_samplesBatch);
  util::DeleteLayerWeights(d_transposeScratch);
}

__global__ void initialiseLayerWeights(LayerWeights layer, const float initRange, Random rnd) {
  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= layer.layerSize || col >= layer.inputSize) {
    return;
  }

  float *out = layer.Elem(row, col);
  *out = row + col; //initRange * (rnd.SampleUniform(col + row * layer.inputSize) * 2.0f - 1.0f);
}

static void initialiseWeights(void) {
  for (auto& lw : d_layerWeights) {
    // Blocks per grid in X and Y dimensions.
    int bpgX = (lw.inputSize + TPB_X - 1) / TPB_X;
    int bpgY = (lw.layerSize + TPB_Y - 1) / TPB_Y;

    float initRange = 1.0f / sqrtf(lw.inputSize);
    initialiseLayerWeights<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1)>>>(lw, initRange, rnd);
  }
}

__global__ void initialiseLayerOutputs(LayerBatchOutputs outputs) {
  const unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= outputs.maxBatchSize) {
    return;
  }

  *(outputs.OutputElem(id, outputs.layerSize - 1)) = 1.0f;
}

static void initialiseOutputs(void) {
  // We initialise the outputs array for each layer to have a 1.0 at the end so that it can
  // be used as the bias input for the next layer.
  for (auto& lo : d_layerOutputs) {
    int bpgX = (lo.maxBatchSize + TPB_X - 1) / TPB_X;
    initialiseLayerOutputs<<<bpgX, TPB_X>>>(lo);
  }
}

void CudaNetwork::Initialise(const NetworkSpec &spec) {
  rnd = Random::Create(2048, 1337);

  networkSpec = spec;
  assert(networkSpec.hiddenActivation != LayerActivation::SOFTMAX);

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
    assert(outWeights[i].rows == d_layerWeights[i].layerSize);
    assert(outWeights[i].cols == d_layerWeights[i].inputSize);

    cudaError_t err = cudaMemcpy2D(
        outWeights[i].data, outWeights[i].cols * sizeof(float), // dst
        d_layerWeights[i].weights, d_layerWeights[i].pitch, // src
        outWeights[i].cols * sizeof(float), outWeights[i].rows, // width, height
        cudaMemcpyDeviceToHost);

    CheckError(err);
  }
}

static void uploadSamplesBatch(const math::MatrixView &batchInputs,
                              const math::MatrixView &batchOutputs);
static void forwardPass(void);
static void backwardPass(void);
static void generateLayerDeltas(void);
static void generateGradient(void);

void CudaNetwork::Train(const math::MatrixView &batchInputs, const math::MatrixView &batchOutputs) {
    uploadSamplesBatch(batchInputs, batchOutputs);

    forwardPass();
    backwardPass();
}

void uploadSamplesBatch(const math::MatrixView &batchInputs, const math::MatrixView &batchOutputs) {
  assert(batchInputs.rows == batchOutputs.rows);
  assert(batchInputs.rows <= d_samplesBatch.maxBatchSize);
  assert(batchInputs.cols == d_samplesBatch.inputDim);
  assert(batchOutputs.cols == d_samplesBatch.targetOutputDim);

  d_samplesBatch.batchSize = batchInputs.rows;

  cudaError_t err = cudaMemcpy2D(
      d_samplesBatch.input, d_samplesBatch.ipitch, // dst
      batchInputs.data, batchInputs.cols * sizeof(float), // src
      batchInputs.cols * sizeof(float), batchInputs.rows, // width, height
      cudaMemcpyHostToDevice);
  CheckError(err);

  err = cudaMemcpy2D(
      d_samplesBatch.targetOutput, d_samplesBatch.opitch, // dst
      batchOutputs.data, batchOutputs.cols * sizeof(float), // src
      batchOutputs.cols * sizeof(float), batchOutputs.rows, // width, height
      cudaMemcpyHostToDevice);
  CheckError(err);
}

void forwardPass(void) {
  for (auto& lo : d_layerOutputs) {
    lo.batchSize = d_samplesBatch.batchSize;
  }

  // copy the batch inputs into the first layer outputs.
  cudaError_t err = cudaMemcpy2D(
      d_layerOutputs[0].output, d_layerOutputs[0].opitch, // dst
      d_samplesBatch.input, d_samplesBatch.ipitch,        // src
      d_samplesBatch.inputDim * sizeof(float), d_samplesBatch.batchSize, // width, height
      cudaMemcpyDeviceToDevice);
  CheckError(err);

  for (unsigned i = 1; i < d_layerOutputs.size(); i++) {
    LayerActivation activation = (i == d_layerOutputs.size() - 1) ?
        networkSpec.outputActivation : networkSpec.hiddenActivation;

    ForwardPassKernel::Apply(d_layerWeights[i-1], d_layerOutputs[i-1], d_layerOutputs[i], activation);
  }

  LayerBatchOutputs lastLayer = d_layerOutputs[d_layerOutputs.size() - 1];
  if (networkSpec.outputActivation == LayerActivation::SOFTMAX) {
    SoftmaxKernel::Apply(lastLayer);
  }

  math::MatrixView output = math::MatrixView::Create(lastLayer.batchSize, lastLayer.layerSize);

  err = cudaMemcpy2D(
      output.data, output.cols * sizeof(float), // dst
      lastLayer.output, lastLayer.opitch, // src
      output.cols * sizeof(float), output.rows, // width, height
      cudaMemcpyDeviceToHost);
  CheckError(err);

  for (unsigned r = 0; r < output.rows; r++) {
    for (unsigned c = 0; c < output.cols; c++) {
      cout << output.data[c + r * output.cols] << "\t";
    }
    cout << endl;
  }
  cout << endl;
}

void backwardPass(void) {
  generateLayerDeltas();
  generateGradient();
}

__global__ void lastLayerDeltasKernel(LayerBatchOutputs networkOutput, SamplesBatch samples,
                                      LayerBatchDeltas out) {
  assert(networkOutput.layerSize == samples.targetOutputDim + 1);
  assert(out.layerSize == samples.targetOutputDim);

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= out.batchSize || col >= out.layerSize) {
    return;
  }

  *out.Elem(row, col) = *networkOutput.OutputElem(row, col) - *samples.TargetOutputElem(row, col);
}

void generateLayerDeltas(void) {
  for (auto& ld : d_layerDeltas) {
    ld.batchSize = d_samplesBatch.batchSize;
  }

  LayerBatchDeltas lastLayerDeltas = d_layerDeltas[d_layerDeltas.size() - 1];
  LayerBatchOutputs networkOutput = d_layerOutputs[d_layerOutputs.size() - 1];

  int bpgX = (lastLayerDeltas.layerSize + TPB_X - 1) / TPB_X;
  int bpgY = (lastLayerDeltas.batchSize + TPB_Y - 1) / TPB_Y;

  lastLayerDeltasKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1)>>>(
      networkOutput, d_samplesBatch, lastLayerDeltas);

  for (int i = d_layerDeltas.size() - 2; i >= 0; i--) {
    LayerWeights transposedWeights;
    transposedWeights.inputSize = d_layerWeights[i + 1].layerSize;
    transposedWeights.layerSize = d_layerWeights[i + 1].inputSize;
    transposedWeights.weights = d_transposeScratch.weights;
    transposedWeights.pitch = d_transposeScratch.pitch;

    TransposeKernel::Apply(d_layerWeights[i + 1], transposedWeights);
    BackwardDeltaKernel::Apply(d_layerDeltas[i + 1], transposedWeights,d_layerDeltas[i]);
  }
}

void generateGradient(void) {

}
