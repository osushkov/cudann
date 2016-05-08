#pragma once

#include "MatrixView.hpp"
#include <Eigen/Dense>

typedef Eigen::VectorXf EVector;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EMatrix;

namespace math {

static inline MatrixView GetMatrixView(EMatrix &m) {
  MatrixView result;
  result.rows = m.rows();
  result.cols = m.cols();
  result.data = m.data();
  return result;
}
}
