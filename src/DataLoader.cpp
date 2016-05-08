
#include "DataLoader.hpp"

#include "image/CharImage.hpp"
#include "image/IdxImages.hpp"
#include "image/IdxLabels.hpp"
#include "image/ImageGenerator.hpp"

#include <map>
#include <string>
#include <vector>

using namespace neuralnetwork;

// Number of images to generate using rotation and translation from each canonical training image.
static constexpr unsigned NUM_DERIVED_IMAGES = 5;

static constexpr float GENERATED_IMAGE_SHIFT_X = 0.1f;
static constexpr float GENERATED_IMAGE_SHIFT_Y = 0.1f;
static constexpr float GENERATED_IMAGE_ROT_THETA = 10.0f * M_PI / 180.0f;
static constexpr float PIXEL_DROPOUT_RATE = 0.0f;

static const ImageGenerator imageGenerator(GENERATED_IMAGE_SHIFT_X, GENERATED_IMAGE_SHIFT_Y,
                                           GENERATED_IMAGE_ROT_THETA, PIXEL_DROPOUT_RATE);

static map<int, vector<CharImage>> loadLabeledImages(string imagePath, string labelPath);

static map<int, vector<CharImage>>
generateDerivedImages(const map<int, vector<CharImage>> &labeledImages, string outDirectory,
                      unsigned numDerived);

static TrainingSample sampleFromCharImage(int label, const CharImage &img);

// Loads the training samples from the given digits files. Optionally generates additional derived
// images from the canonical loaded samples.
vector<TrainingSample> DataLoader::LoadSamples(string inImagePath, string inLabelPath,
                                               bool genDerived) {
  auto labeledImages = loadLabeledImages(inImagePath, inLabelPath);

  if (genDerived) {
    labeledImages = generateDerivedImages(labeledImages, "data/images/", NUM_DERIVED_IMAGES);
  }

  vector<TrainingSample> result;
  int inputSize = 0;
  int outputSize = 0;

  for (const auto &entry : labeledImages) {
    for (const auto &image : entry.second) {
      result.push_back(sampleFromCharImage(entry.first, image));
      inputSize = result.back().input.rows();
      outputSize = result.back().expectedOutput.rows();
    }
  }

  for (const auto &sample : result) {
    assert(inputSize > 0 && outputSize > 0);
    assert(sample.input.rows() == inputSize);
    assert(sample.expectedOutput.rows() == outputSize);
  }

  return result;
}

map<int, vector<CharImage>> loadLabeledImages(string imagePath, string labelPath) {
  IdxImages imageLoader(imagePath);
  IdxLabels labelLoader(labelPath);

  vector<int> labels = labelLoader.Load();
  vector<CharImage> images = imageLoader.Load();

  assert(labels.size() == images.size());

  map<int, vector<CharImage>> result;
  for (unsigned i = 0; i < labels.size(); i++) {
    if (result.find(labels[i]) == result.end()) {
      result[labels[i]] = vector<CharImage>();
    }

    result[labels[i]].push_back(images[i]);
  }

  return result;
}

map<int, vector<CharImage>> generateDerivedImages(const map<int, vector<CharImage>> &labeledImages,
                                                  string outDirectory, unsigned numDerived) {

  assert(numDerived >= 1);
  map<int, vector<CharImage>> result;

  for (const auto &entry : labeledImages) {
    int digit = entry.first;

    result[digit] = vector<CharImage>();
    result[digit].reserve(entry.second.size() * numDerived);

    for (const auto &image : entry.second) {
      vector<CharImage> generated = imageGenerator.GenerateImages(image, numDerived);

      for (auto &gimage : generated) {
        assert(gimage.width == image.width && gimage.height == image.height);
        result[digit].push_back(gimage);
      }
    }
  }

  return result;
}

TrainingSample sampleFromCharImage(int label, const CharImage &img) {
  EVector output(10);
  output.fill(0.0f);
  output[label] = 1.0f;

  EVector input(img.pixels.size());
  for (unsigned i = 0; i < img.pixels.size(); i++) {
    input(i) = img.pixels[i];
  }

  return neuralnetwork::TrainingSample(input, output);
}
