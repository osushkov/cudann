
#pragma once

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

SamplesBatch NewSamplesBatch(unsigned batchSize, unsigned sampleDim);
void DeleteSamplesBatch(SamplesBatch &sb);

LayerBatchOutputs NewLayerBatchOutputs(unsigned batchSize, unsigned layerSize);
void DeleteLayerBatchOutputs(LayerBatchOutputs &lbo);

LayerBatchDeltas NewLayerBatchDeltas(unsigned batchSize, unsigned layerSize);
void DeleteLayerBatchDeltas(LayerBatchDeltas &lbd);
}
}
}