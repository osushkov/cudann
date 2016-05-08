#pragma once

#include "CharImage.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

class ImageGenerator {
  float shiftX;
  float shiftY;
  float rotTheta;
  float dropoutRate;

public:
  ImageGenerator(float shiftX, float shiftY, float rotTheta, float dropoutRate);

  vector<CharImage> GenerateImages(const CharImage &base, unsigned numImages) const;

private:
  struct Transform {
    float tx, ty; // translation in pixels
    float theta;  // rotation in radians

    Transform(float tx, float ty, float theta) : tx(tx), ty(ty), theta(theta) {}
  };

  Transform randomTransform(const CharImage &img) const;

  cv::Mat convertToMat(const CharImage &img) const;
  CharImage convertToCharImage(const cv::Mat &img) const;

  CharImage transformToCharImage(const cv::Mat &src, const Transform &transform) const;
};
