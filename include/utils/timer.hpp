#pragma once

#include <chrono>

namespace anns
{
  namespace utils
  {

    class Timer
    {
    private:
      using clock = std::chrono::high_resolution_clock;
      using time_point = std::chrono::time_point<clock>;
      time_point t1_;
      double total_;

    public:
      /// @brief A tiny timer to test runtime in "second" unit.
      Timer() : total_(0) {}

      void reset() { total_ = 0; }

      void start() { t1_ = clock::now(); }

      void stop()
      {
        total_ += (std::chrono::duration<double, std::milli>(clock::now() - t1_).count() / 1000);
      }

      double get() { return total_; }
    };

  }
}