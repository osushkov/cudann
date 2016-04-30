
#pragma once

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
    return (float *)((char *)weights + r * pitch) + c;
  }
};

struct SamplesBatch {
  unsigned batchSize; // equal to the number of rows in the matrix.
  unsigned sampleDim; // equal to the number of columns in the matrix.

  float *input; // matrix sized numSamples(rows) * sampleDim(cols)
  size_t pitch;

  __device__ float *Elem(unsigned r, unsigned c) {
    return (float *)((char *)input + r * pitch) + c;
  }
};

struct LayerBatchOutputs {
  unsigned batchSize;
  unsigned layerSize;

  float *output; // matrix sized batchSize(rows) * layerSize(cols)
  size_t opitch;

  float *derivative; // matrix sized batchSize(rows) * layerSize(cols)
  size_t dpitch;

  __device__ float *OutputElem(unsigned r, unsigned c) {
    return (float *)((char *)output + r * opitch) + c;
  }

  __device__ float *DerivativeElem(unsigned r, unsigned c) {
    return (float *)((char *)derivative + r * dpitch) + c;
  }
};

struct LayerBatchDeltas {
  unsigned batchSize;
  unsigned layerSize;

  float *delta; // matrix sized batchSize(rows) * layerSize(cols)
  size_t pitch;

  __device__ float *Elem(unsigned r, unsigned c) {
    return (float *)((char *)delta + r * pitch) + c;
  }
};
}
}
