#pragma once

#include "../common/Common.hpp"
#include "CharImage.hpp"
#include <iostream>
#include <string>
#include <vector>

class IdxImages {
  string filePath;

public:
  IdxImages(string filePath);

  vector<CharImage> Load(void) const;

private:
  bool loadAndCheckMagicNumber(istream &in) const;
  int loadNumEntries(istream &in) const;
  CharImage loadImage(istream &in, int width, int height) const;
};
