
#pragma once

#include "../../math/MatrixView.hpp"
#include "Types.hpp"
#include <cuda_runtime.h>

#define CheckError(ans)                                                                            \
  { neuralnetwork::cuda::util::OutputError((ans), __FILE__, __LINE__); }

namespace neuralnetwork {
namespace cuda {
namespace util {

void OutputError(cudaError_t code, const char *file, int line);

void *AllocPushBuffer(size_t bufSize);
void FreePushBuffer(void *buf);

LayerWeights NewLayerWeights(unsigned inputSize, unsigned layerSize);
void DeleteLayerWeights(LayerWeights &lw);

SamplesBatch NewSamplesBatch(unsigned maxBatchSize, unsigned inputDim, unsigned targetOutputDim);
void DeleteSamplesBatch(SamplesBatch &sb);

LayerBatchOutputs NewLayerBatchOutputs(unsigned maxBatchSize, unsigned layerSize);
void DeleteLayerBatchOutputs(LayerBatchOutputs &lbo);

LayerBatchDeltas NewLayerBatchDeltas(unsigned maxBatchSize, unsigned layerSize);
void DeleteLayerBatchDeltas(LayerBatchDeltas &lbd);

void PrintMatrixView(math::MatrixView view);
void PrintLayerWeights(LayerWeights d_weights);
void PrintLayerOutputs(LayerBatchOutputs d_outputs);
void PrintLayerDeltas(LayerBatchDeltas d_deltas);
}
}
}
