#pragma once

#include "../common/Common.hpp"
#include "Math.hpp"

namespace math {

class Tensor {
public:
  unsigned NumLayers(void) const;
  void AddLayer(const EMatrix &m);

  EMatrix &operator()(unsigned index);
  const EMatrix &operator()(unsigned index) const;

  Tensor operator*(const Tensor &t) const;
  Tensor operator+(const Tensor &t) const;
  Tensor operator-(const Tensor &t) const;
  Tensor operator*(float s) const;
  Tensor operator/(float s) const;

  Tensor &operator*=(const Tensor &t);
  Tensor &operator+=(const Tensor &t);
  Tensor &operator-=(const Tensor &t);
  Tensor &operator*=(float s);
  Tensor &operator/=(float s);

private:
  vector<EMatrix> data;
};
}
