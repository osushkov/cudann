
#include "CudaNetwork.hpp"
#include "Util.hpp"
#include "Random.hpp"

#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdio>

#include <curand.h>
#include <cuda_runtime.h>

// The default num threads per 2D block.
static constexpr int TPB_X = 2;
static constexpr int TPB_Y = 2;

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
    d_layerGradients.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
    d_layerOutputs.push_back(util::NewLayerBatchOutputs(networkSpec.maxBatchSize, layerSizes[i] + 1));
    d_layerDeltas.push_back(util::NewLayerBatchDeltas(networkSpec.maxBatchSize, layerSizes[i]));
  }

  d_samplesBatch = util::NewSamplesBatch(networkSpec.maxBatchSize, networkSpec.numInputs);
}

static void freeDeviceMemory(void) {
  for (auto& lw : d_layerWeights) { util::DeleteLayerWeights(lw); }
  for (auto& lg : d_layerGradients) { util::DeleteLayerWeights(lg); }
  for (auto& lo : d_layerOutputs) { util::DeleteLayerBatchOutputs(lo); }
  for (auto& ld : d_layerDeltas) { util::DeleteLayerBatchDeltas(ld); }
  util::DeleteSamplesBatch(d_samplesBatch);
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

static void forwardPass(const math::MatrixView &batchInputs);
static void applySoftmax(const LayerBatchOutputs &lastLayer);
// static void backwardPass(const math::MatrixView &batchOutputs);

void CudaNetwork::Train(const math::MatrixView &batchInputs, const math::MatrixView &batchOutputs) {
    forwardPass(batchInputs);
}

__device__ float activationValue(float in, const LayerActivation activation) {
  switch(activation) {
  case LayerActivation::TANH:
    return tanhf(in);
  case LayerActivation::LOGISTIC:
    return 1.0f / (1.0f + expf(-in));
  case LayerActivation::RELU:
    return fmaxf(0.0f, in);
  case LayerActivation::LEAKY_RELU:
    return fmaxf(0.01f * in, in);
  case LayerActivation::LINEAR:
  case LayerActivation::SOFTMAX:
    return in;
  }
  assert(false); // should never get here.
  return in;
}

__device__ float activationDerivative(float in, float out, const LayerActivation activation) {
  switch(activation) {
  case LayerActivation::TANH:
    return 1.0f - out * out;
  case LayerActivation::LOGISTIC:
    return out * (1.0f - out);
  case LayerActivation::RELU:
    return in > 0.0f ? 1.0f : 0.0f;
  case LayerActivation::LEAKY_RELU:
    return in > 0.0f ? 1.0f : 0.01f;
  case LayerActivation::LINEAR:
  case LayerActivation::SOFTMAX:
    return 1.0f;
  }
  assert(false); // should never get here.
  return 1.0f;
}

__global__ void layerForwardPass(LayerWeights lw, LayerBatchOutputs prevOutputs,
                                 LayerBatchOutputs out, const LayerActivation activation) {
  assert(lw.inputSize == prevOutputs.layerSize);
  assert(lw.layerSize == out.layerSize - 1);

  extern __shared__ float buf[]; // shared memory buffer

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  // TODO: can implement a "fast path" and "slow path" versions of the below code and branch here.
  // Fast path can assume that the entire block will fall within the bounds of all of the matrices
  // and dispense with a whole bunch of the below checks.

  const int numChunks = (lw.inputSize + blockDim.x - 1) / blockDim.x;

  // buffer for holding the layer weight matrix chunk
  float *lwChunk = (float *) buf;

  // buffer for holding the prev outputs matrix chunk
  float *poChunk = (float *) &buf[blockDim.x * blockDim.y];

  float sum = 0.0f;
  const int lwRow = blockDim.x * blockIdx.x + threadIdx.y;
  const int poRow = row;

  const int chunkIndex = threadIdx.x + threadIdx.y * blockDim.x;
  const int lim = numChunks * blockDim.x;

  for (int chunkOffset = 0; chunkOffset < lim; chunkOffset += blockDim.x) {
    const int lwCol = chunkOffset + threadIdx.x;
    if (lwRow < lw.layerSize && lwCol < lw.inputSize) {
      lwChunk[chunkIndex] = *lw.Elem(lwRow, lwCol);
    }

    const int poCol = lwCol;
    if (poRow < prevOutputs.batchSize && poCol < prevOutputs.layerSize) {
      poChunk[chunkIndex] = *prevOutputs.OutputElem(poRow, poCol);
    }
    __syncthreads();

    int chunkLim = min(blockDim.x, lw.inputSize - chunkOffset);
    for (int j = 0; j < chunkLim; j++) {
      sum += lwChunk[j + threadIdx.x * blockDim.x] * poChunk[j + threadIdx.y * blockDim.x];
    }
    __syncthreads();
  }

  if (row < out.batchSize && col < out.layerSize - 1) {
    float *outElem = out.OutputElem(row, col);
    *outElem = activationValue(sum, activation);

    float *dElem = out.DerivativeElem(row, col);
    *dElem = activationDerivative(sum, *outElem, activation);
  }
}

void forwardPass(const math::MatrixView &batchInputs) {
  for (auto& lo : d_layerOutputs) {
    assert(batchInputs.rows <= lo.maxBatchSize);
    lo.batchSize = batchInputs.rows;
  }

  // copy the batch inputs into the first layer outputs.
  cudaError_t err = cudaMemcpy2D(
      d_layerOutputs[0].output, d_layerOutputs[0].opitch, // dst
      batchInputs.data, batchInputs.cols * sizeof(float), // src
      batchInputs.cols * sizeof(float), batchInputs.rows, // width, height
      cudaMemcpyHostToDevice);
  CheckError(err);

  for (unsigned i = 1; i < d_layerOutputs.size(); i++) {
    // -1 is here since we dont need to compute the bias term for the output vector.
    int bpgX = (d_layerOutputs[i].layerSize - 1 + TPB_X - 1) / TPB_X;
    int bpgY = (d_layerOutputs[i].batchSize + TPB_Y - 1) / TPB_Y;
    size_t sharedMemSize = 2 * TPB_X * TPB_Y * sizeof(float);

    LayerActivation activation = (i == d_layerOutputs.size() - 1) ?
        networkSpec.outputActivation : networkSpec.hiddenActivation;

    layerForwardPass<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize>>>(
        d_layerWeights[i-1], d_layerOutputs[i-1], d_layerOutputs[i], activation);
  }

  LayerBatchOutputs lastLayer = d_layerOutputs[d_layerOutputs.size() - 1];
  if (networkSpec.outputActivation == LayerActivation::SOFTMAX) {
    applySoftmax(lastLayer);
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

// This softmax code assumes that the output layer is smaller than the maximum number of threads
// in a block. For ease of implementation, we assume this and do the whole thing in a single block.
// This allows easy synchronization and easy algorithm. Most problems wont have >1024 outputs.
// Separate blocks can do separate batch rows.
__global__ void layerForwardPass(LayerBatchOutputs outputs) {
  extern __shared__ float buf[]; // shared memory buffer

  const unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
  // if (index )
}

void applySoftmax(const LayerBatchOutputs &lastLayer) {
  // assert(lastLayer.)

}
