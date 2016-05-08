
#include "IdxLabels.hpp"
#include "IdxUtil.hpp"
#include <fstream>
#include <iostream>

static constexpr int IDX_MAGIC_NUM = 2049;

IdxLabels::IdxLabels(string filePath) : filePath(filePath) {}

vector<int> IdxLabels::Load(void) const {
  vector<int> result;

  std::ifstream is(filePath, std::ifstream::binary);
  if (is) {
    if (!loadAndCheckMagicNumber(is)) {
      cout << "Error, invalid file magic number" << endl;
      return result;
    }

    int numLabels = loadNumEntries(is);

    result.reserve(numLabels);
    for (int i = 0; i < numLabels; i++) {
      int label = static_cast<int>(Idx::readUchar(is));
      assert(label >= 0 && label <= 9);

      result.push_back(label);
    }
  } else {
    cout << "Error, no such file: " << filePath << endl;
  }

  return result;
}

bool IdxLabels::loadAndCheckMagicNumber(istream &in) const {
  return Idx::readInt(in) == IDX_MAGIC_NUM;
}

int IdxLabels::loadNumEntries(istream &in) const { return Idx::readInt(in); }
