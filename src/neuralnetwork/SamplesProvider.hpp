
#pragma once

#include "../common/Common.hpp"
#include "TrainingSample.hpp"
#include <cassert>
#include <vector>

namespace neuralnetwork {

class SamplesProvider {
public:
  SamplesProvider(const vector<TrainingSample> &allSamples)
      : SamplesProvider(allSamples, allSamples.size(), 0) {}

  SamplesProvider(const vector<TrainingSample> &allSamples, unsigned numSamples, unsigned offset)
      : allSamples(allSamples), numSamples(numSamples), offset(offset) {}

  const TrainingSample &operator[](unsigned index) const {
    assert(index < numSamples);
    unsigned i = (index + offset) % allSamples.size();
    return allSamples[i];
  }

  unsigned NumSamples(void) const { return numSamples; }

private:
  const vector<TrainingSample> &allSamples;
  unsigned numSamples;
  unsigned offset;
};
}
