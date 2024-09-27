#pragma once

#include <vector>
#include <memory>

namespace anns
{
  namespace interval
  {

    template <typename T> class IntervalFilter {
     public:
      std::unique_ptr<std::vector<std::pair<T, T>>> intervals_;
      size_t num_points_;

      IntervalFilter(size_t num_points) noexcept: num_points_(num_points)  {
        intervals_ = std::make_unique<std::vector<std::pair<T, T>>>(num_points);
      }

      void 

    };

  } // namespace interval
  
} // namespace anns
