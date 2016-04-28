#pragma once

#include "../math/Math.hpp"

namespace neuralnetwork {

struct TrainingSample {
  EVector input;
  EVector expectedOutput;

  TrainingSample(const EVector &input, const EVector &expectedOutput)
      : input(input), expectedOutput(expectedOutput) {}
};
}
