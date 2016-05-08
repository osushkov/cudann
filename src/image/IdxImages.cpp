
#include "IdxImages.hpp"
#include "IdxUtil.hpp"
#include <fstream>
#include <iostream>

static constexpr int IDX_MAGIC_NUM = 2051;

IdxImages::IdxImages(string filePath) : filePath(filePath) {}

vector<CharImage> IdxImages::Load(void) const {
  vector<CharImage> result;

  std::ifstream is(filePath, std::ifstream::binary);
  if (is) {
    if (!loadAndCheckMagicNumber(is)) {
      cout << "Error, invalid file magic number" << endl;
      return result;
    }

    int numImages = loadNumEntries(is);

    int imgWidth = Idx::readInt(is);
    int imgHeight = Idx::readInt(is);
    assert(imgWidth > 0 && imgHeight > 0);

    result.reserve(numImages);
    for (int i = 0; i < numImages; i++) {
      result.push_back(loadImage(is, imgWidth, imgHeight));
    }
  } else {
    cout << "Error, no such file: " << filePath << endl;
  }

  return result;
}

bool IdxImages::loadAndCheckMagicNumber(istream &in) const {
  return Idx::readInt(in) == IDX_MAGIC_NUM;
}

int IdxImages::loadNumEntries(istream &in) const { return Idx::readInt(in); }

CharImage IdxImages::loadImage(istream &in, int width, int height) const {
  assert(width > 0 && height > 0);

  vector<float> pixels;
  pixels.reserve(width * height);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float pixelValue = Idx::readUchar(in) / 255.0f;
      pixels.push_back(pixelValue);
    }
  }

  return CharImage(width, height, pixels);
}
