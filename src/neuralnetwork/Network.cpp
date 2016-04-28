
#include "Network.hpp"
#include "../math/Tensor.hpp"
#include "Activations.hpp"
#include <cassert>

using namespace neuralnetwork;

struct Network::NetworkImpl {
  NetworkSpec spec;
  math::Tensor layerWeights;

  NetworkImpl(const NetworkSpec &spec) : spec(spec) {
    assert(spec.numInputs > 0 && spec.numOutputs > 0);
  }

  EVector Process(const EVector &input) {
    assert(input.rows() == spec.numInputs);

    return input;
  }

  void Update(const SamplesProvider &samplesProvider) {}

  // Returns the output vector of the layer, and the derivative vector for the layer.
  pair<EVector, EVector> getLayerOutput(const EVector &prevLayer, const EMatrix &layerWeights,
                                        LayerActivation afunc) const {
    EVector z = layerWeights * getInputWithBias(prevLayer);
    // layerWeights.topRightCorner(layerWeights.rows(), layerWeights.cols() - 1) * prevLayer;
    EVector derivatives(z.rows());

    // if (isOutput && useSoftmax) {
    //   z = Util::SoftmaxActivations(z);
    // } else {
    //   for (unsigned i = 0; i < layerWeights.rows(); i++) {
    //     // z(i) += layerWeights(i, 0);
    //     float in = z(i);
    //
    //     z(i) = func->ActivationValue(in);
    //     derivatives(i) = func->DerivativeValue(in, z(i));
    //   }
    // }

    return make_pair(z, derivatives);
  }

  EVector getInputWithBias(const EVector &noBiasInput) const {
    EVector result(noBiasInput.rows() + 1);
    result(noBiasInput.rows()) = 1.0f;
    result.topRightCorner(noBiasInput.rows(), 1) = noBiasInput;
    return result;
  }
};

Network::Network(const NetworkSpec &spec) : impl(new NetworkImpl(spec)) {}
Network::~Network() = default;

EVector Network::Process(const EVector &input) { return impl->Process(input); }
void Network::Update(const SamplesProvider &samplesProvider) { impl->Update(samplesProvider); }
