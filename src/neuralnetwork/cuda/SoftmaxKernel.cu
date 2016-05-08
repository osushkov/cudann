
#include "SoftmaxKernel.hpp"
#include <cuda_runtime.h>

using namespace neuralnetwork::cuda;

// This softmax code assumes that the output layer is smaller than the maximum number of threads
// in a block. For ease of implementation, we assume this and do the whole thing in a single block.
// This allows easy synchronization and easy algorithm. Most problems wont have >1024 outputs.
// Separate blocks can do separate batch rows.
__global__ void softmaxKernel(LayerBatchOutputs outputs) {
  extern __shared__ float buf[]; // shared memory buffer

  const unsigned outIndex = threadIdx.x;
  const unsigned batchIndex = blockIdx.x;

  assert(blockDim.x <= outputs.layerSize && gridDim.x == outputs.batchSize);

  // A single float to hold data to exchange between threads in this block.
  float *sharedVar = (float *) &buf[0];

  // Buffer to hold all of the output elements for this batch element.
  float *outElems = (float *) &buf[1];

  // 1. Copy the row for the current batch into shared memory.
  float val = *(outputs.OutputElem(batchIndex, outIndex));
  outElems[outIndex] = val;
  __syncthreads();

  // 2. Find the max element in the row, done by a single thread per block while all others wait.
  float maxValue;
  if (outIndex == 0) {
    maxValue = outElems[0];
    for (unsigned i = 1; i < blockDim.x; i++) {
      maxValue = fmaxf(maxValue, outElems[i]);
    }
    *sharedVar = maxValue;
  }
  __syncthreads();
  maxValue = *sharedVar;

  // 3. Calc the unnormalised exponent offset by the max value and write it to shared mem.
  val = expf(val - maxValue);
  outElems[outIndex] = val;
  __syncthreads();

  // 4. Calculate the sum across the batch, done by a single thread per block.
  float sum = 0.0f;
  if (outIndex == 0) {
    for (unsigned i = 0; i < blockDim.x; i++) {
      sum += outElems[i];
    }
    *sharedVar = sum;
  }
  __syncthreads();
  sum = *sharedVar;

  // 5. Calc the normalised value for each output elem and write it out to global mem.
  *(outputs.OutputElem(batchIndex, outIndex)) = val / sum;
}

void SoftmaxKernel::Apply(const LayerBatchOutputs &lastLayer, cudaStream_t stream) {
  size_t sharedMemSize = (lastLayer.layerSize + 1) * sizeof(float);

  // We dont want to include the bias part of the output in the processing of the softmax.
  int tpb = lastLayer.layerSize - 1;
  int bpg = lastLayer.batchSize;
  softmaxKernel<<<bpg, tpb, sharedMemSize, stream>>>(lastLayer);
}
