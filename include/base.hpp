#pragma once

#include <anns.hpp>
#include <string>
#include <atomic>

namespace anns
{

  template <typename data_t>
  class Base
  {
  protected:
    DataSet<data_t> base_;
    size_t num_threads_{1};
    std::atomic<size_t> comparison_{0};

  public:
    virtual void build(const DataSet<data_t> &base) = 0;

    /// By define save function, you can define your child graph with the constructor:
    virtual void save(const std::string &filename) const = 0;

    virtual res_t search(const DataSet<data_t> &query, size_t k, size_t ef) = 0;

    virtual std::priority_queue<std::pair<float, int>> search(const data_t *data_point, size_t k, size_t ef) = 0;

    virtual size_t index_size() const = 0;

    size_t get_comparison_and_clear() noexcept
    {
      return comparison_.exchange(0);
    }

    size_t get_num_threads() const noexcept
    {
      return num_threads_;
    }

    void set_num_threads(size_t num_threads) noexcept
    {
      num_threads_ = num_threads;
    }
  };

#define __USE_BASE__                            \    
  using Base<data_t>::base_;                    \
  using Base<data_t>::num_threads_;             \
  using Base<data_t>::comparison_;              \
  using Base<data_t>::get_comparison_and_clear; \
  using Base<data_t>::get_num_threads;          \
  using Base<data_t>::set_num_threads;

}