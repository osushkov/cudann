
#pragma once

#include <cassert>

namespace math {

struct MatrixView {
  unsigned rows;
  unsigned cols;
  float *data; // row major order.

  static MatrixView Create(unsigned rows, unsigned cols) {
    assert(rows > 0 && cols > 0);

    MatrixView result;
    result.rows = rows;
    result.cols = cols;
    result.data = new float[rows * cols];
    return result;
  }

  static void Release(MatrixView &view) {
    assert(view.data != nullptr);
    delete[] view.data;
    view.data = nullptr;
  }
};
}
