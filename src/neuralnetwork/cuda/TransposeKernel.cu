
#include "TransposeKernel.hpp"
#include "Constants.hpp"
#include <cuda_runtime.h>

using namespace neuralnetwork::cuda;

__global__ void transposeKernel(LayerWeights lw, LayerWeights out, unsigned bufStride) {
  extern __shared__ float buf[];

  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < lw.inputSize && y < lw.layerSize) {
    buf[threadIdx.x + threadIdx.y * bufStride] = *lw.Elem(y, x);
  }

  __syncthreads();

  x = blockIdx.y * blockDim.y + threadIdx.x;  // transpose block offset
  y = blockIdx.x * blockDim.x + threadIdx.y;

  if (x < out.inputSize && y < out.layerSize) {
    *(out.Elem(y, x)) = buf[threadIdx.y + threadIdx.x * bufStride];
  }
}

void TransposeKernel::Apply(LayerWeights layerWeights, LayerWeights transposedWeights,
                            cudaStream_t stream) {
  int bpgX = (layerWeights.inputSize + TPB_X - 1) / TPB_X;
  int bpgY = (layerWeights.layerSize + TPB_Y - 1) / TPB_Y;

  unsigned stride = TPB_X + 1;
  size_t sharedMemSize = stride * TPB_Y * sizeof(float);

  transposeKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      layerWeights, transposedWeights, stride);
}
