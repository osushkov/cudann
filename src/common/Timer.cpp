
#include "Timer.hpp"

static const unsigned MICRO_SECONDS_IN_SECOND = 1000000;

void Timer::Start(void) { gettimeofday(&startTime, NULL); }

void Timer::Stop(void) { gettimeofday(&endTime, NULL); }

float Timer::GetNumElapsedSeconds(void) const {
  unsigned seconds = endTime.tv_sec - startTime.tv_sec;
  unsigned mseconds = seconds * MICRO_SECONDS_IN_SECOND - startTime.tv_usec + endTime.tv_usec;

  return (float)mseconds / (float)MICRO_SECONDS_IN_SECOND;
}

unsigned Timer::GetNumElapsedMicroseconds(void) const {
  unsigned seconds = endTime.tv_sec - startTime.tv_sec;
  return seconds * MICRO_SECONDS_IN_SECOND - startTime.tv_usec + endTime.tv_usec;
}
