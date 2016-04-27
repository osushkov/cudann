
#pragma once

#include <sys/time.h>
#include <time.h>

class Timer {
public:
  void Start(void);
  void Stop(void);
  float GetNumElapsedSeconds(void) const;
  unsigned GetNumElapsedMicroseconds(void) const;

private:
  struct timeval startTime;
  struct timeval endTime;
};
