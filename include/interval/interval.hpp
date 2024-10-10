#pragma once

#include <utility>
#include <vector>
#include <algorithm>
#include <cassert>

namespace anns
{
  namespace interval
  {
    
    template <typename label_t> 
    class Interval: public std::pair<label_t, label_t>
    {
    public:

      Interval() = default;

      Interval(const Interval<label_t>& other) = default;

      Interval(std::initializer_list<label_t> obj) 
      {
        assert(obj.size() == 2 && "Initializer list must have exactly 2 elements");
        auto it = obj.begin();
        this->first = *it;
        this->second = *(++it);
      }

      static inline bool cover(const Interval<label_t>& a, const Interval<label_t>& b) noexcept
      {
        return a.first <= b.first && a.second >= b.second;
      }

      inline bool cover(const Interval<label_t>& b) const noexcept
      {
        return this->first <= b.first && this->second >= b.second;
      }

      static inline bool overlap(const Interval<label_t>& a, const Interval<label_t>& b) noexcept
      {
        return std::max(a.first, b.first) <= std::min(a.second, b.second);
      }

      inline bool overlap(const Interval<label_t>& b) const noexcept
      {
        return std::max(this->first, b.first) <= std::min(this->second, b.second);
      }

    };

#define I_COVER (i1, i2) i1.cover(i2)
#define I_OVERLAP (i1, i2) i1.overlap(i2)

  } 
  
}
