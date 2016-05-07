
#include "Network.hpp"
#include "../common/Util.hpp"
#include "../math/Tensor.hpp"
#include "Activations.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>

using namespace neuralnetwork;

struct Network::NetworkImpl {
  NetworkSpec spec;
  math::Tensor layerWeights;

  NetworkImpl(const NetworkSpec &spec) : spec(spec) {
    assert(spec.numInputs > 0 && spec.numOutputs > 0);
    initialiseWeights();
  }

  EVector Process(const EVector &input) {
    assert(input.rows() == spec.numInputs);

    EVector layerOutput = input;
    for (unsigned i = 0; i < layerWeights.NumLayers(); i++) {
      LayerActivation func =
          (i == layerWeights.NumLayers() - 1) ? spec.outputActivation : spec.hiddenActivation;
      layerOutput = getLayerOutput(layerOutput, layerWeights(i), func);
    }

    return layerOutput;
  }

  void Update(const SamplesProvider &samplesProvider) {}

  EVector getLayerOutput(const EVector &prevLayer, const EMatrix &layerWeights,
                         LayerActivation afunc) const {
    assert(prevLayer.rows() == layerWeights.cols() - 1);

    EVector z = layerWeights * getInputWithBias(prevLayer);
    if (afunc == LayerActivation::SOFTMAX) {
      z = softmaxActivations(z);
    } else {
      for (unsigned i = 0; i < z.rows(); i++) {
        z(i) = ActivationValue(afunc, z(i));
      }
    }

    return z;
  }

  EVector getInputWithBias(const EVector &noBiasInput) const {
    EVector result(noBiasInput.rows() + 1);
    result(noBiasInput.rows()) = 1.0f;
    result.topRightCorner(noBiasInput.rows(), 1) = noBiasInput;
    return result;
  }

  void initialiseWeights(void) {
    if (spec.hiddenLayers.empty()) {
      layerWeights.AddLayer(createLayer(spec.numInputs, spec.numOutputs));
    } else {
      layerWeights.AddLayer(createLayer(spec.numInputs, spec.hiddenLayers[0]));

      for (unsigned i = 0; i < spec.hiddenLayers.size() - 1; i++) {
        layerWeights.AddLayer(createLayer(spec.hiddenLayers[i], spec.hiddenLayers[i + 1]));
      }

      layerWeights.AddLayer(
          createLayer(spec.hiddenLayers[spec.hiddenLayers.size() - 1], spec.numOutputs));
    }
  }

  EMatrix createLayer(unsigned inputSize, unsigned layerSize) const {
    assert(inputSize > 0 && layerSize > 0);

    unsigned numRows = layerSize;
    unsigned numCols = inputSize + 1; // +1 accounts for bias input
    float initRange = 1.0f / sqrtf(numCols);

    EMatrix result(numRows, numCols);
    for (unsigned r = 0; r < result.rows(); r++) {
      for (unsigned c = 0; c < result.cols(); c++) {
        result(r, c) = Util::RandInterval(-initRange, initRange);
      }
    }

    return result;
  }

  EVector softmaxActivations(const EVector &in) const {
    assert(in.rows() > 0);
    EVector result(in.rows());

    float maxVal = in(0);
    for (int r = 0; r < in.rows(); r++) {
      maxVal = fmax(maxVal, in(r));
    }

    float sum = 0.0f;
    for (int i = 0; i < in.rows(); i++) {
      result(i) = expf(in(i)-maxVal);
      sum += result(i);
    }

    for (int i = 0; i < result.rows(); i++) {
      result(i) /= sum;
    }

    return result;
  }
};

Network::Network(const NetworkSpec &spec) : impl(new NetworkImpl(spec)) {}
Network::~Network() = default;

EVector Network::Process(const EVector &input) { return impl->Process(input); }
void Network::Update(const SamplesProvider &samplesProvider) { impl->Update(samplesProvider); }
