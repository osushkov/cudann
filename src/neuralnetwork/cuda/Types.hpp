
#pragma once

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>

namespace neuralnetwork {
namespace cuda {

struct LayerWeights {
  unsigned inputSize; // this includes the bias. So it should be equal to prev layer size + 1
  unsigned layerSize;

  // Data pointers allocated with cudaMallocPitch. Logical size is (inputSize * layerSize)
  // num rows = layerSize, num cols = inputSize
  float *weights;

  // The pitch of the rows of the weights matrix in bytes.
  size_t pitch;

  __device__ float *Elem(unsigned r, unsigned c) {
    assert(r < layerSize && c < inputSize);
    return (float *)((char *)weights + r * pitch) + c;
  }
};

struct SamplesBatch {
  unsigned maxBatchSize; // number of rows allocated in memory.
  unsigned batchSize;    // equal to the number of rows in the matrix.
  unsigned inputDim;     // equal to the number of columns in the matrix.
  unsigned targetOutputDim;

  float *input; // matrix sized batchSize(rows) * sampleDim(cols)
  size_t ipitch;

  float *targetOutput; // matrix sized batchSize(rows) * sampleDim(cols)
  size_t opitch;

  __device__ float *InputElem(unsigned r, unsigned c) {
    assert(r < maxBatchSize && c < inputDim);
    return (float *)((char *)input + r * ipitch) + c;
  }

  __device__ float *TargetOutputElem(unsigned r, unsigned c) {
    assert(r < maxBatchSize && c < targetOutputDim);
    return (float *)((char *)targetOutput + r * opitch) + c;
  }
};

struct LayerBatchOutputs {
  unsigned maxBatchSize;
  unsigned batchSize;

  // layer size includes the bias term, so it will be equal to the number of nodes + 1
  unsigned layerSize;

  float *output; // matrix sized batchSize(rows) * layerSize(cols)
  size_t opitch;

  float *derivative; // matrix sized batchSize(rows) * layerSize(cols)
  size_t dpitch;

  __device__ float *OutputElem(unsigned r, unsigned c) {
    assert(r < maxBatchSize && c < layerSize);
    return (float *)((char *)output + r * opitch) + c;
  }

  __device__ float *DerivativeElem(unsigned r, unsigned c) {
    assert(r < maxBatchSize && c < layerSize);
    return (float *)((char *)derivative + r * dpitch) + c;
  }
};

struct LayerBatchDeltas {
  unsigned maxBatchSize;
  unsigned batchSize;
  unsigned layerSize;

  float *delta; // matrix sized batchSize(rows) * layerSize(cols)
  size_t pitch;

  __device__ float *Elem(unsigned r, unsigned c) {
    assert(r < maxBatchSize && c < layerSize);
    return (float *)((char *)delta + r * pitch) + c;
  }
};
}
}
