#pragma once

#include "../common/Common.hpp"
#include <iostream>
#include <string>
#include <vector>

class IdxLabels {
  string filePath;

public:
  IdxLabels(string filePath);

  vector<int> Load(void) const;

private:
  bool loadAndCheckMagicNumber(istream &in) const;
  int loadNumEntries(istream &in) const;
};
