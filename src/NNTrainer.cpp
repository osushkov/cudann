
#include "NNTrainer.hpp"

#include "DataLoader.hpp"
#include "ExperimentUtils.hpp"
#include "common/Common.hpp"
#include "common/Timer.hpp"
#include "common/Util.hpp"
#include "neuralnetwork/Network.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace neuralnetwork;

static unsigned numCompletePasses;
static unsigned curSamplesIndex;
static unsigned curSamplesOffset;

static uptr<Network> createNewNetwork(unsigned inputSize, unsigned outputSize);

static void learn(Network *network, vector<TrainingSample> &trainingSamples,
                  vector<TrainingSample> &testSamples);

static SamplesProvider getStochasticSamples(vector<TrainingSample> &allSamples, unsigned curIter,
                                            unsigned totalIters);

void NNTrainer::TrainAndEvaluate(void) {
  string trainImagesPath = "/home/osushkov/Programming/cudann/data/train_images.idx3";
  string trainLabelsPath = "/home/osushkov/Programming/cudann/data/train_labels.idx1";

  string evalImagesPath = "/home/osushkov/Programming/cudann/data/test_images.idx3";
  string evalLabelsPath = "/home/osushkov/Programming/cudann/data/test_labels.idx1";

  cout << "loading training data" << endl;
  vector<TrainingSample> trainingSamples =
      DataLoader::LoadSamples(trainImagesPath, trainLabelsPath, true);
  random_shuffle(trainingSamples.begin(), trainingSamples.end());
  cout << "training data size: " << trainingSamples.size() << endl;

  cout << "loading test data" << endl;
  vector<TrainingSample> testSamples =
      DataLoader::LoadSamples(evalImagesPath, evalLabelsPath, false);
  random_shuffle(testSamples.begin(), testSamples.end());
  cout << "test data size: " << testSamples.size() << endl;

  unsigned inputSize = trainingSamples.front().input.rows();
  unsigned outputSize = trainingSamples.front().expectedOutput.rows();

  auto network = createNewNetwork(inputSize, outputSize);
  learn(network.get(), trainingSamples, testSamples);
}

uptr<Network> createNewNetwork(unsigned inputSize, unsigned outputSize) {
  NetworkSpec spec;
  spec.numInputs = inputSize;
  spec.numOutputs = outputSize;
  spec.hiddenLayers = {inputSize, inputSize / 2};
  spec.nodeActivationRate = 0.6f;
  spec.maxBatchSize = 2000;
  spec.hiddenActivation = LayerActivation::LEAKY_RELU;
  spec.outputActivation = LayerActivation::SOFTMAX;

  return make_unique<Network>(spec);
}

void learn(Network *network, vector<TrainingSample> &trainingSamples,
           vector<TrainingSample> &testSamples) {

  numCompletePasses = 0;
  curSamplesIndex = 0;
  curSamplesOffset = 0;

  cout << "starting training..." << endl;
  Timer timer;
  timer.Start();

  const unsigned iters = 20000;
  for (unsigned i = 0; i < iters; i++) {
    auto samplesProvider = getStochasticSamples(trainingSamples, i, iters);
    network->Update(samplesProvider);

    if (i % 100 == 0) {
      network->Refresh();
      float testWrong = ExperimentUtils::Eval(network, testSamples);
      cout << i << "\t" << testWrong << endl;
    }
  }
  timer.Stop();
  cout << "finished" << endl;
  cout << "elapsed seconds: " << timer.GetNumElapsedSeconds() << endl;
}

SamplesProvider getStochasticSamples(vector<TrainingSample> &allSamples, unsigned curIter,
                                     unsigned totalIters) {

  unsigned numSamples = min<unsigned>(allSamples.size(), 2000);

  if ((curSamplesIndex + numSamples) > allSamples.size()) {
    if (numCompletePasses % 10 == 0) {
      random_shuffle(allSamples.begin(), allSamples.end());
    } else {
      curSamplesOffset = rand() % allSamples.size();
    }
    curSamplesIndex = 0;
    numCompletePasses++;
  }

  auto result = SamplesProvider(allSamples, numSamples, curSamplesIndex + curSamplesOffset);
  curSamplesIndex += numSamples;

  return result;
}
