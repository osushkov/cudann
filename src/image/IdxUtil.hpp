#pragma once

#include <cassert>
#include <iostream>

namespace Idx {

inline int readInt(istream &in) {
  unsigned char bigEndianBuf[4];
  in.read((char *)bigEndianBuf, 4);

  int result = 0;
  assert(sizeof(result) == 4); // or like, use int32 or something here.

  unsigned char *littleEndianBuf = (unsigned char *)&result;
  littleEndianBuf[0] = bigEndianBuf[3];
  littleEndianBuf[1] = bigEndianBuf[2];
  littleEndianBuf[2] = bigEndianBuf[1];
  littleEndianBuf[3] = bigEndianBuf[0];

  return result;
}

inline unsigned char readUchar(istream &in) {
  unsigned char result = 0;
  in.read((char *)&result, sizeof(unsigned char));
  return result;
}
}
