
#include "ForwardPassKernel.hpp"
#include "Constants.hpp"
#include <cuda_runtime.h>

using namespace neuralnetwork;
using namespace neuralnetwork::cuda;

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

__global__ void forwardPassKernel(LayerWeights lw, LayerBatchOutputs prevOutputs,
                                  LayerBatchOutputs out, const LayerActivation activation,
                                  Random rnd, const float nodeActivationRate, const bool isOutput,
                                  const unsigned spitch) {

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
  float *poChunk = (float *) &buf[spitch * blockDim.y];

  float sum = 0.0f;
  const int lwRow = blockDim.x * blockIdx.x + threadIdx.y;
  const int poRow = row;

  const int chunkIndex = threadIdx.x + threadIdx.y * spitch;
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
      sum += lwChunk[j + threadIdx.x * spitch] * poChunk[j + threadIdx.y * spitch];
    }
    __syncthreads();
  }

  if (row < out.batchSize && col < out.layerSize - 1) {
    float *outElem = out.OutputElem(row, col);
    float *dElem = out.DerivativeElem(row, col);

    if (isOutput || rnd.SampleUniform(col + row * out.layerSize) < nodeActivationRate) {
      *outElem = activationValue(sum, activation);
      *dElem = activationDerivative(sum, *outElem, activation);
    } else {
      *outElem = 0.0f;
      *dElem = 0.0f;
    }
  }
}

void ForwardPassKernel::Apply(LayerWeights layerWeights, LayerBatchOutputs input,
                              LayerBatchOutputs output, LayerActivation activation,
                              Random rnd, float nodeActivationRate, bool isOutputLayer,
                              cudaStream_t stream) {
  assert(layerWeights.inputSize == input.layerSize);
  assert(layerWeights.layerSize == output.layerSize - 1);

  // -1 is here since we dont need to compute the bias term for the output vector.
  int bpgX = (output.layerSize - 1 + TPB_X - 1) / TPB_X;
  int bpgY = (output.batchSize + TPB_Y - 1) / TPB_Y;

  unsigned spitch = (TPB_X + 1);
  size_t sharedMemSize = 2 * spitch * TPB_Y * sizeof(float);

  forwardPassKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      layerWeights, input, output, activation, rnd, nodeActivationRate, isOutputLayer, spitch);
}
