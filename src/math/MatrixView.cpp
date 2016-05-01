
#include "MatrixView.hpp"
#include <algorithm>
#include <cassert>

using namespace math;

MatrixView MatrixView::Create(unsigned rows, unsigned cols) {
  assert(rows > 0 && cols > 0);

  MatrixView result;
  result.rows = rows;
  result.cols = cols;
  result.data = new float[rows * cols];
  return result;
}

MatrixView MatrixView::CreateZeroed(unsigned rows, unsigned cols) {
  MatrixView result = Create(rows, cols);
  std::fill(result.data, result.data + rows * cols, 0.0f);
  return result;
}

void MatrixView::Release(MatrixView &view) {
  assert(view.data != nullptr);
  delete[] view.data;
  view.data = nullptr;
}
