#include "GradientKernel.hpp"
#include "Constants.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

using namespace neuralnetwork;
using namespace neuralnetwork::cuda;

__global__ void gradientKernel(LayerBatchDeltas layerDeltas, LayerBatchOutputs layerOutputs,
                               LayerWeights outGradient, unsigned spitch) {
  extern __shared__ float buf[]; // shared memory buffer

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  // buffer for holding the layer weight matrix chunk
  float *ldChunk = (float *) buf;

  // buffer for holding the prev outputs matrix chunk
  float *loChunk = (float *) &buf[spitch * blockDim.y];

  const int ldCol = blockDim.y * blockIdx.y + threadIdx.x;
  const int loCol = col;

  const int numChunks = (layerDeltas.batchSize + blockDim.y - 1) / blockDim.y;
  const int chunkIndex = threadIdx.x + threadIdx.y * spitch;
  const int lim = numChunks * blockDim.y;

  float sum = 0.0f;
  for (int chunkOffset = 0; chunkOffset < lim; chunkOffset += blockDim.y) {
    const int ldRow = chunkOffset + threadIdx.y;
    if (ldRow < layerDeltas.batchSize && ldCol < layerDeltas.layerSize) {
      ldChunk[chunkIndex] = *layerDeltas.Elem(ldRow, ldCol);
    }

    const int loRow = ldRow;
    if (loRow < layerOutputs.batchSize && loCol < layerOutputs.layerSize) {
      loChunk[chunkIndex] = *layerOutputs.OutputElem(loRow, loCol);
    }
    __syncthreads();

    int chunkLim = min(blockDim.x, layerDeltas.batchSize - chunkOffset);
    for (int j = 0; j < chunkLim; j++) {
      sum += ldChunk[threadIdx.y + j * spitch] * loChunk[threadIdx.x + j * spitch];
    }

    __syncthreads();
  }

  if (row < outGradient.layerSize && col < outGradient.inputSize) {
    *outGradient.Elem(row, col) = sum / layerDeltas.batchSize;
  }
}

void GradientKernel::Apply(LayerBatchDeltas layerDeltas, LayerBatchOutputs layerOutputs,
                           LayerWeights outGradient, cudaStream_t stream) {

  assert(layerDeltas.batchSize == layerOutputs.batchSize);
  assert(layerDeltas.layerSize == outGradient.layerSize);
  assert(layerOutputs.layerSize == outGradient.inputSize);

  int bpgX = (outGradient.inputSize + TPB_X - 1) / TPB_X;
  int bpgY = (outGradient.layerSize + TPB_Y - 1) / TPB_Y;

  unsigned spitch = (TPB_X + 1);
  size_t sharedMemSize = 2 * spitch * TPB_Y * sizeof(float);

  gradientKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      layerDeltas, layerOutputs, outGradient, spitch);
}
