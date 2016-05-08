
#include "BackwardDeltaKernel.hpp"
#include "Constants.hpp"
#include <cuda_runtime.h>

using namespace neuralnetwork;
using namespace neuralnetwork::cuda;


// computes outDelta = tw * nextDelta (elemwisemul) layerOutput.derivatives
__global__ void backwardDeltaKernel(LayerBatchDeltas nextDelta, LayerWeights tw,
                                    LayerBatchOutputs layerOutput, LayerBatchDeltas outDelta,
                                    unsigned spitch) {

  extern __shared__ float buf[]; // shared memory buffer

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  const int numChunks = (tw.inputSize + blockDim.x - 1) / blockDim.x;

  // buffer for holding the layer weight matrix chunk
  float *twChunk = (float *) buf;

  // buffer for holding the prev outputs matrix chunk
  float *ndChunk = (float *) &buf[spitch * blockDim.y];

  const int twRow = blockDim.x * blockIdx.x + threadIdx.y;
  const int ndRow = row;

  const int chunkIndex = threadIdx.x + threadIdx.y * spitch;
  const int lim = numChunks * blockDim.x;

  float sum = 0.0f;
  for (int chunkOffset = 0; chunkOffset < lim; chunkOffset += blockDim.x) {
    const int twCol = chunkOffset + threadIdx.x;
    if (twRow < tw.layerSize && twCol < tw.inputSize) {
      twChunk[chunkIndex] = *tw.Elem(twRow, twCol);
    }

    const int ndCol = twCol;
    if (ndRow < nextDelta.batchSize && ndCol < nextDelta.layerSize) {
      ndChunk[chunkIndex] = *nextDelta.Elem(ndRow, ndCol);
    }
    __syncthreads();

    int chunkLim = min(blockDim.x, tw.inputSize - chunkOffset);
    for (int j = 0; j < chunkLim; j++) {
      sum += twChunk[j + threadIdx.x * spitch] * ndChunk[j + threadIdx.y * spitch];
    }
    __syncthreads();
  }

  if (row < outDelta.batchSize && col < outDelta.layerSize) {
    float od = *layerOutput.DerivativeElem(row, col);
    *outDelta.Elem(row, col) = sum * od;
  }
}

void BackwardDeltaKernel::Apply(LayerBatchDeltas nextDelta, LayerWeights transposedWeights,
                                LayerBatchOutputs layerOutput, LayerBatchDeltas outDelta,
                                cudaStream_t stream) {

  // TODO: handle bank conflicts. Do the same in the forward kernel.
  assert(nextDelta.layerSize == transposedWeights.inputSize);
  assert(outDelta.layerSize == transposedWeights.layerSize - 1);
  assert(outDelta.layerSize == layerOutput.layerSize - 1);
  assert(nextDelta.batchSize == layerOutput.batchSize);
  assert(nextDelta.batchSize == outDelta.batchSize);

  int bpgX = (outDelta.layerSize + TPB_X - 1) / TPB_X;
  int bpgY = (outDelta.batchSize + TPB_Y - 1) / TPB_Y;

  unsigned spitch = (TPB_X + 1);
  size_t sharedMemSize = 2 * spitch * TPB_Y * sizeof(float);

  backwardDeltaKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      nextDelta, transposedWeights, layerOutput, outDelta, spitch);
}
