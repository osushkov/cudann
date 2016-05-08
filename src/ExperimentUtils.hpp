
#pragma once

#include "common/Common.hpp"
#include "neuralnetwork/Network.hpp"
#include "neuralnetwork/TrainingSample.hpp"
#include <vector>

namespace ExperimentUtils {
float Eval(neuralnetwork::Network *network,
           const vector<neuralnetwork::TrainingSample> &testSamples);
}
