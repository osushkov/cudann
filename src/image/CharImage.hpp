#pragma once

#include "../common/Common.hpp"
#include <cassert>
#include <vector>

struct CharImage {
  unsigned width;
  unsigned height;

  // TODO: this can be an std::array since mnist data is 28x28 images.
  vector<float> pixels; // row wise

  CharImage(unsigned width, unsigned height, const vector<float> &pixels)
      : width(width), height(height), pixels(pixels) {
    assert(width * height == pixels.size());
  }
};
