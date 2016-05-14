
#include "CudaNetwork.hpp"
#include "Util.hpp"
#include "Random.hpp"
#include "SoftmaxKernel.hpp"
#include "ForwardPassKernel.hpp"
#include "TransposeKernel.hpp"
#include "BackwardDeltaKernel.hpp"
#include "GradientKernel.hpp"
#include "Constants.hpp"

#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdio>
#include <mutex>

#include <curand.h>
#include <cuda_runtime.h>

using namespace neuralnetwork;
using namespace neuralnetwork::cuda;
using namespace std;

// ADAM trainer parameters
static constexpr float adamBeta1 = 0.9f;
static constexpr float adamBeta2 = 0.999f;
static constexpr float adamEpsilon = 10e-8;
static constexpr float adamLearnRate = 0.001f;

static Random rnd;
static std::once_flag stateFlag;

static void initialiseSharedState(void) {
  std::call_once(stateFlag, [](){
    rnd = Random::Create(2048, 1337);
  });
}

__global__ void initialiseLayerWeights(LayerWeights layer, const float initRange, Random rnd) {
  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= layer.layerSize || col >= layer.inputSize) {
    return;
  }

  float *out = layer.Elem(row, col);
  *out = initRange * (rnd.SampleUniform(col + row * layer.inputSize) * 2.0f - 1.0f);
}

__global__ void initialiseLayerOutputs(LayerBatchOutputs outputs) {
  const unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= outputs.maxBatchSize) {
    return;
  }

  *(outputs.OutputElem(id, outputs.layerSize - 1)) = 1.0f;
}

__global__ void initialiseAdamWeights(LayerWeights momentum, LayerWeights rms) {
  assert(momentum.inputSize == rms.inputSize);
  assert(momentum.layerSize == rms.layerSize);

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= rms.layerSize || col >= rms.inputSize) {
    return;
  }

  *momentum.Elem(row, col) = 0.0f;
  *rms.Elem(row, col) = 0.0f;
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

  // TODO: check whether reading into shared mem, doing computation, then writing to global mem
  // is faster. You never know.
  *out.Elem(row, col) = *networkOutput.OutputElem(row, col) - *samples.TargetOutputElem(row, col);
}

__global__ void updateMomentumAndRMS(LayerWeights gradient, LayerWeights momentum, LayerWeights rms,
                                      const float beta1, const float beta2) {
  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= gradient.layerSize || col >= gradient.inputSize) {
    return;
  }

  float g = *gradient.Elem(row, col);
  float m = *momentum.Elem(row, col);
  float r = *rms.Elem(row, col);

  *momentum.Elem(row, col) = m * beta1 + g * (1.0f - beta1);
  *rms.Elem(row, col) = r * beta2 + g * g * (1.0f - beta2);
}

__global__ void updateWeightsWithAdam(LayerWeights weights, LayerWeights momentum, LayerWeights rms,
                                      const float beta1, const float beta2,
                                      const float lr, const float epsilon) {

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= rms.layerSize || col >= rms.inputSize) {
    return;
  }

  float mc = *momentum.Elem(row, col) / (1.0f - beta1);
  float rc = *rms.Elem(row, col) / (1.0f - beta2);

  *weights.Elem(row, col) -= lr * mc / sqrtf(rc + epsilon);
}

struct CudaNetwork::CudaNetworkImpl {
  NetworkSpec networkSpec;
  vector<LayerWeights> d_layerWeights;
  vector<LayerWeights> d_layerWeightsBridge;
  vector<LayerWeights> d_layerGradients;
  vector<LayerBatchOutputs> d_layerOutputs;
  vector<LayerBatchDeltas> d_layerDeltas;
  SamplesBatch d_samplesBatch;

  LayerWeights d_transposeScratch;

  // TODO: this stuff should go into a separate file. Trainer code/variables should be
  // separate from network code.
  vector<LayerWeights> d_adamMomentum;
  vector<LayerWeights> d_adamRMS;

  cudaStream_t uploadStream;
  cudaStream_t computeStream;

  CudaNetworkImpl(const NetworkSpec &spec) : networkSpec(spec) {
    assert(networkSpec.hiddenActivation != LayerActivation::SOFTMAX);
    initialiseSharedState();

    uploadStream = 0;
    computeStream = 0;

    allocDeviceMemory();
    initialiseWeights();
    initialiseOutputs();
    initialiseADAM();
  }

  ~CudaNetworkImpl() {
    for (auto& lw : d_layerWeights) { util::DeleteLayerWeights(lw); }
    for (auto& lw : d_layerWeightsBridge) { util::DeleteLayerWeights(lw); }
    for (auto& lg : d_layerGradients) { util::DeleteLayerWeights(lg); }
    for (auto& lo : d_layerOutputs) { util::DeleteLayerBatchOutputs(lo); }
    for (auto& ld : d_layerDeltas) { util::DeleteLayerBatchDeltas(ld); }
    for (auto& am : d_adamMomentum) { util::DeleteLayerWeights(am); }
    for (auto& am : d_adamRMS) { util::DeleteLayerWeights(am); }
    util::DeleteSamplesBatch(d_samplesBatch);
    util::DeleteLayerWeights(d_transposeScratch);
  }

  void SetWeights(const std::vector<math::MatrixView> &weights) {
    assert(d_layerWeights.size() == weights.size());

    for (unsigned i = 0; i < weights.size(); i++) {
      assert(weights[i].rows == d_layerWeights[i].layerSize);
      assert(weights[i].cols == d_layerWeights[i].inputSize);

      cudaError_t err = cudaMemcpy2D(
          d_layerWeights[i].weights, d_layerWeights[i].pitch,
          weights[i].data, weights[i].cols * sizeof(float),
          weights[i].cols * sizeof(float), weights[i].rows,
          cudaMemcpyHostToDevice);

      CheckError(err);
    }
  }

  void GetWeights(std::vector<math::MatrixView> &outWeights) {
    assert(outWeights.size() == d_layerWeightsBridge.size());

    for (unsigned i = 0; i < outWeights.size(); i++) {
      assert(outWeights[i].rows == d_layerWeightsBridge[i].layerSize);
      assert(outWeights[i].cols == d_layerWeightsBridge[i].inputSize);

      cudaError_t err = cudaMemcpy2DAsync(
          outWeights[i].data, outWeights[i].cols * sizeof(float), // dst
          d_layerWeightsBridge[i].weights, d_layerWeightsBridge[i].pitch, // src
          outWeights[i].cols * sizeof(float), outWeights[i].rows, // width, height
          cudaMemcpyDeviceToHost, uploadStream);

      CheckError(err);
    }
  }

  void Train(const math::MatrixView &batchInputs, const math::MatrixView &batchOutputs) {
    uploadSamplesBatch(batchInputs, batchOutputs);

    forwardPass();
    backwardPass();
    updateAdamParams();
    updateWeights();

    for (unsigned i = 0; i < d_layerWeights.size(); i++) {
      cudaError_t err = cudaMemcpy2D(
          d_layerWeightsBridge[i].weights, d_layerWeightsBridge[i].pitch,
          d_layerWeights[i].weights, d_layerWeights[i].pitch,
          d_layerWeights[i].inputSize * sizeof(float), d_layerWeights[i].layerSize,
          cudaMemcpyDeviceToDevice);

      CheckError(err);
    }
  }

private:
  void uploadSamplesBatch(const math::MatrixView &batchInputs,
                          const math::MatrixView &batchOutputs) {
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
    cudaError_t err = cudaMemcpy2DAsync(
        d_layerOutputs[0].output, d_layerOutputs[0].opitch, // dst
        d_samplesBatch.input, d_samplesBatch.ipitch,        // src
        d_samplesBatch.inputDim * sizeof(float), d_samplesBatch.batchSize, // width, height
        cudaMemcpyDeviceToDevice, computeStream);
    CheckError(err);

    for (unsigned i = 1; i < d_layerOutputs.size(); i++) {
      LayerActivation activation = (i == d_layerOutputs.size() - 1) ?
          networkSpec.outputActivation : networkSpec.hiddenActivation;

      ForwardPassKernel::Apply(d_layerWeights[i-1], d_layerOutputs[i-1], d_layerOutputs[i],
          activation, rnd, networkSpec.nodeActivationRate, i == (d_layerOutputs.size() - 1),
          computeStream);
    }

    LayerBatchOutputs lastLayer = d_layerOutputs[d_layerOutputs.size() - 1];
    if (networkSpec.outputActivation == LayerActivation::SOFTMAX) {
      SoftmaxKernel::Apply(lastLayer, computeStream);
    }
  }

  void backwardPass(void) {
    generateLayerDeltas();
    generateGradient();
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

      TransposeKernel::Apply(d_layerWeights[i + 1], transposedWeights, computeStream);

      BackwardDeltaKernel::Apply(d_layerDeltas[i + 1], transposedWeights, d_layerOutputs[i+1],
                                 d_layerDeltas[i], computeStream);
    }
  }

  void generateGradient(void) {
    for (unsigned i = 0; i < d_layerWeights.size(); i++) {
      GradientKernel::Apply(d_layerDeltas[i], d_layerOutputs[i], d_layerGradients[i], computeStream);
    }
  }

  void updateAdamParams(void) {
    for (unsigned i = 0; i < d_layerGradients.size(); i++) {
      int bpgX = (d_layerGradients[i].inputSize + TPB_X - 1) / TPB_X;
      int bpgY = (d_layerGradients[i].layerSize + TPB_Y - 1) / TPB_Y;

      updateMomentumAndRMS<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1)>>>(
          d_layerGradients[i], d_adamMomentum[i], d_adamRMS[i], adamBeta1, adamBeta2);
    }
  }

  void updateWeights(void) {
    for (unsigned i = 0; i < d_layerWeights.size(); i++) {
      int bpgX = (d_layerWeights[i].inputSize + TPB_X - 1) / TPB_X;
      int bpgY = (d_layerWeights[i].layerSize + TPB_Y - 1) / TPB_Y;

      updateWeightsWithAdam<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1)>>>(
          d_layerWeights[i], d_adamMomentum[i], d_adamRMS[i],
          adamBeta1, adamBeta2, adamLearnRate, adamEpsilon);
    }
  }

  void initialiseADAM(void) {
    assert(d_adamRMS.size() == d_adamMomentum.size());

    for (unsigned i = 0; i < d_adamRMS.size(); i++) {
      int bpgX = (d_adamRMS[i].inputSize + TPB_X - 1) / TPB_X;
      int bpgY = (d_adamRMS[i].layerSize + TPB_Y - 1) / TPB_Y;

      initialiseAdamWeights<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1)>>>(
          d_adamMomentum[i], d_adamRMS[i]);
    }
  }

  void initialiseOutputs(void) {
    // We initialise the outputs array for each layer to have a 1.0 at the end so that it can
    // be used as the bias input for the next layer.
    for (auto& lo : d_layerOutputs) {
      int bpgX = (lo.maxBatchSize + TPB_X - 1) / TPB_X;
      initialiseLayerOutputs<<<bpgX, TPB_X>>>(lo);
    }
  }

  void initialiseWeights(void) {
    for (auto& lw : d_layerWeights) {
      // Blocks per grid in X and Y dimensions.
      int bpgX = (lw.inputSize + TPB_X - 1) / TPB_X;
      int bpgY = (lw.layerSize + TPB_Y - 1) / TPB_Y;

      float initRange = 1.0f / sqrtf(lw.inputSize);
      initialiseLayerWeights<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1)>>>(lw, initRange, rnd);
    }
  }

  // Pre-allocated all of the device memory we will need. We should never have to malloc device
  // memory after this function is called.
  void allocDeviceMemory(void) {
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
      d_layerWeightsBridge.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
      d_layerGradients.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
      d_layerOutputs.push_back(util::NewLayerBatchOutputs(networkSpec.maxBatchSize, layerSizes[i] + 1));
      d_layerDeltas.push_back(util::NewLayerBatchDeltas(networkSpec.maxBatchSize, layerSizes[i]));

      d_adamMomentum.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
      d_adamRMS.push_back(util::NewLayerWeights(prevLayerSize + 1, layerSizes[i]));
    }

    d_samplesBatch =
        util::NewSamplesBatch(networkSpec.maxBatchSize, networkSpec.numInputs, networkSpec.numOutputs);

    d_transposeScratch = util::NewLayerWeights(maxLayerSize, maxInputSize);
  }
};


CudaNetwork::CudaNetwork(const NetworkSpec &spec) : impl(new CudaNetworkImpl(spec)) {}
CudaNetwork::~CudaNetwork() = default;

void CudaNetwork::SetWeights(const std::vector<math::MatrixView> &weights) {
    impl->SetWeights(weights);
}

void CudaNetwork::GetWeights(std::vector<math::MatrixView> &outWeights) {
  impl->GetWeights(outWeights);
}

void CudaNetwork::Train(const math::MatrixView &batchInputs, const math::MatrixView &batchOutputs) {
  impl->Train(batchInputs, batchOutputs);
}
