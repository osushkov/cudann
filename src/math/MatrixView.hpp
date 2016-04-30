
#pragma once

namespace math {

struct MatrixView {
  unsigned rows;
  unsigned cols;
  float *data; // row major order.

  static MatrixView Create(unsigned rows, unsigned cols);
  static MatrixView CreateZeroed(unsigned rows, unsigned cols);
  static void Release(MatrixView &view);
};
}
