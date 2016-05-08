#pragma once

#include "common/Common.hpp"
#include "neuralnetwork/TrainingSample.hpp"
#include <vector>

namespace DataLoader {
vector<neuralnetwork::TrainingSample> LoadSamples(string inImagePath, string inLabelPath,
                                                  bool genDerived);
}
